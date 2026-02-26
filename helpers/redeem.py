"""
Gasless redemption of winning Polymarket positions via Builder Relayer.
No MATIC needed — the relayer pays gas.

Usage:
    from helpers.redeem import init_relayer, redeem_market, check_token_balance

    relayer = init_relayer(private_key, api_key, api_secret, passphrase)
    success = redeem_market(relayer, condition_id, neg_risk=False)
"""
import os
import time
from datetime import datetime, timezone

from web3 import Web3
from eth_abi import encode

from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import SafeTransaction, OperationType

from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

# ─── Contract Addresses (Polygon) ────────────────────────────────────────
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
RELAYER_URL = "https://relayer-v2.polymarket.com"


def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [REDEEM] {msg}")


def init_relayer(private_key: str, api_key: str, api_secret: str, passphrase: str) -> RelayClient:
    """Initialize the Builder Relayer client."""
    creds = BuilderApiKeyCreds(key=api_key, secret=api_secret, passphrase=passphrase)
    builder_config = BuilderConfig(local_builder_creds=creds)
    relayer = RelayClient(
        relayer_url=RELAYER_URL,
        chain_id=137,
        private_key=private_key,
        builder_config=builder_config,
    )
    return relayer


def check_token_balance(clob_client, token_id: str) -> float:
    """Check conditional token balance via CLOB API. Returns amount (raw_units / 1e6)."""
    try:
        params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id=token_id,
            signature_type=2,
        )
        ba = clob_client.get_balance_allowance(params)
        raw = int(ba.get("balance", "0"))
        return raw / 1e6
    except Exception:
        return 0.0


def redeem_market(relayer: RelayClient, condition_id: str, neg_risk: bool = False) -> bool:
    """Redeem resolved positions via gasless relayer.

    Args:
        relayer: Initialized RelayClient
        condition_id: Market condition ID (hex string)
        neg_risk: True for neg-risk markets (NegRiskAdapter), False for standard CTF

    Returns:
        True if redemption confirmed on-chain, False otherwise.
    """
    # Ensure condition_id is bytes32
    if not condition_id.startswith("0x"):
        condition_id = "0x" + condition_id
    cond_bytes = bytes.fromhex(condition_id[2:].zfill(64))

    if neg_risk:
        # NegRiskAdapter.redeemPositions(bytes32 conditionId, uint256[] amounts)
        selector = Web3.keccak(text="redeemPositions(bytes32,uint256[])")[:4]
        max_uint = 2**256 - 1
        params = encode(["bytes32", "uint256[]"], [cond_bytes, [max_uint, max_uint]])
        target = NEG_RISK_ADAPTER
    else:
        # CTF.redeemPositions(address collateral, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)
        selector = Web3.keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
        params = encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [USDC, b"\x00" * 32, cond_bytes, [1, 2]],
        )
        target = CTF

    calldata = "0x" + (selector + params).hex()

    tx = SafeTransaction(
        to=target,
        operation=OperationType.Call,
        data=calldata,
        value="0",
    )

    _log(f"Submitting redeem tx (neg_risk={neg_risk})...")
    resp = relayer.execute([tx], "Redeem positions")
    tx_id = resp.transaction_id
    _log(f"Transaction ID: {tx_id}")

    # Poll for confirmation (up to 20 x 3s = 60s)
    for i in range(20):
        time.sleep(3)
        status = relayer.get_transaction(tx_id)
        if isinstance(status, list) and len(status) > 0:
            tx_state = status[0].get("state", "")
            tx_hash = status[0].get("transactionHash", "")
            if tx_hash:
                _log(f"  Poll {i+1}: {tx_state} | hash: {tx_hash[:20]}...")
            else:
                _log(f"  Poll {i+1}: {tx_state}")

            if "CONFIRMED" in tx_state:
                _log(f"CONFIRMED! TX: https://polygonscan.com/tx/{tx_hash}")
                return True
            if "FAILED" in tx_state or "INVALID" in tx_state:
                _log(f"FAILED: {tx_state}")
                return False
        else:
            _log(f"  Poll {i+1}: waiting...")

    _log("Timed out waiting for confirmation")
    return False
