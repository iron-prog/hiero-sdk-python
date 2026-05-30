"""Unit tests for the mirror node REST client."""

from __future__ import annotations

import pytest

from hiero_sdk_python import MirrorNodeClient
from hiero_sdk_python.account.account_id import AccountId
from hiero_sdk_python.hbar import Hbar
from hiero_sdk_python.mirror_node import MirrorNodeError
from hiero_sdk_python.mirror_node.mirror_node_client import (
    MirrorNodeAccount,
    MirrorNodeBalance,
    MirrorNodePage,
    MirrorNodeTransaction,
)
from hiero_sdk_python.tokens.token_id import TokenId


pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, timeout, headers):
        self.calls.append({"url": url, "timeout": timeout, "headers": headers})
        if not self.responses:
            raise AssertionError(f"Unexpected request to {url}")
        return self.responses.pop(0)


def account_payload():
    return {
        "account": "0.0.1001",
        "alias": "HIEROALIAS",
        "balance": {
            "balance": 123456789,
            "timestamp": "1710000000.000000001",
            "tokens": [
                {"token_id": "0.0.2002", "balance": 50},
                {"token_id": "0.0.2003", "balance": "75"},
            ],
        },
        "deleted": False,
        "evm_address": "0x00000000000000000000000000000000000003e9",
        "key": {"_type": "ED25519", "key": "abc"},
        "memo": "sample account",
        "receiver_sig_required": True,
        "staking_info": {"decline_reward": False},
    }


def transaction_payload(transaction_id="0.0.1001@1710000000.123456789", consensus="1710000001.000000001"):
    return {
        "transaction_id": transaction_id,
        "consensus_timestamp": consensus,
        "result": "SUCCESS",
        "name": "CRYPTOTRANSFER",
        "charged_tx_fee": 150000,
        "max_fee": 200000000,
        "memo_base64": "SGllcm8=",
        "node": "0.0.3",
    }


def test_for_network_uses_default_mirror_node_base_url():
    client = MirrorNodeClient.for_network("mainnet")

    assert client.base_url == "https://mainnet-public.mirrornode.hedera.com"


def test_custom_network_string_is_treated_as_base_url():
    client = MirrorNodeClient.for_network("http://mirror.local:5551")

    assert client.base_url == "http://mirror.local:5551"


def test_rejects_invalid_base_url_and_timeout():
    with pytest.raises(ValueError, match="base_url"):
        MirrorNodeClient(base_url="mirror.local")

    with pytest.raises(ValueError, match="timeout"):
        MirrorNodeClient(base_url="http://mirror.local", timeout=0)


def test_get_account_builds_request_and_maps_balances():
    session = FakeSession([FakeResponse(payload=account_payload())])
    client = MirrorNodeClient(base_url="http://mirror.local", session=session, timeout=2.5, headers={"x-api-key": "test"})

    account = client.get_account(AccountId(0, 0, 1001), transactions=True)

    assert session.calls == [
        {
            "url": "http://mirror.local/api/v1/accounts/0.0.1001?transactions=true",
            "timeout": 2.5,
            "headers": {"x-api-key": "test"},
        }
    ]
    assert account.account_id == AccountId(0, 0, 1001)
    assert account.alias == "HIEROALIAS"
    assert account.evm_address == "0x00000000000000000000000000000000000003e9"
    assert account.balance.hbars == Hbar.from_tinybars(123456789)
    assert account.balance.tokens[TokenId(0, 0, 2002)] == 50
    assert account.balance.tokens[TokenId(0, 0, 2003)] == 75
    assert account.receiver_sig_required is True
    assert account.raw["memo"] == "sample account"


def test_get_account_balance_returns_zero_when_payload_has_no_balance():
    session = FakeSession([FakeResponse(payload={"account": "0.0.1001"})])
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    balance = client.get_account_balance("0.0.1001")

    assert balance.hbars == Hbar.ZERO
    assert balance.tokens == {}


def test_get_token_maps_common_metadata():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "token_id": "0.0.2002",
                    "name": "Example Token",
                    "symbol": "EXT",
                    "decimals": "8",
                    "total_supply": "1000000000",
                    "treasury_account_id": "0.0.1001",
                    "type": "FUNGIBLE_COMMON",
                    "deleted": False,
                }
            )
        ]
    )
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    token = client.get_token(TokenId(0, 0, 2002))

    assert session.calls[0]["url"] == "http://mirror.local/api/v1/tokens/0.0.2002"
    assert token.token_id == TokenId(0, 0, 2002)
    assert token.name == "Example Token"
    assert token.decimals == 8
    assert token.total_supply == 1000000000
    assert token.treasury_account_id == AccountId(0, 0, 1001)


def test_list_transactions_sends_common_filters_and_maps_page():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "transactions": [transaction_payload()],
                    "links": {"next": "/api/v1/transactions?limit=1&order=asc&timestamp=gt:1"},
                }
            )
        ]
    )
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    page = client.list_transactions(
        account_id=AccountId(0, 0, 1001),
        result="success",
        transaction_type="cryptotransfer",
        order="asc",
        limit=1,
    )

    assert session.calls[0]["url"] == (
        "http://mirror.local/api/v1/transactions?account.id=0.0.1001&result=success"
        "&type=cryptotransfer&order=asc&limit=1"
    )
    assert page.has_next is True
    assert page.next == "/api/v1/transactions?limit=1&order=asc&timestamp=gt:1"
    assert page.items[0].transaction_id.account_id == AccountId(0, 0, 1001)
    assert page.items[0].charged_tx_fee == Hbar.from_tinybars(150000)
    assert page.items[0].node_account_id == AccountId(0, 0, 3)


def test_get_transaction_uses_transaction_id_filter_and_returns_first_match():
    session = FakeSession([FakeResponse(payload={"transactions": [transaction_payload()], "links": {"next": None}})])
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    transaction = client.get_transaction("0.0.1001@1710000000.123456789")

    assert session.calls[0]["url"] == (
        "http://mirror.local/api/v1/transactions?transactionid=0.0.1001%401710000000.123456789&limit=1"
    )
    assert transaction.result == "SUCCESS"


def test_get_transaction_raises_when_no_transaction_matches():
    session = FakeSession([FakeResponse(payload={"transactions": [], "links": {"next": None}})])
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    with pytest.raises(MirrorNodeError, match="No transaction found"):
        client.get_transaction("0.0.1001@1710000000.123456789")


def test_iterate_transactions_follows_next_links():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "transactions": [transaction_payload(consensus="1.000000001")],
                    "links": {"next": "/api/v1/transactions?limit=1&timestamp=gt:1.000000001"},
                }
            ),
            FakeResponse(
                payload={
                    "transactions": [transaction_payload(consensus="2.000000001")],
                    "links": {"next": None},
                }
            ),
        ]
    )
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    transactions = list(client.iterate_transactions(limit=1))

    assert [transaction.consensus_timestamp for transaction in transactions] == ["1.000000001", "2.000000001"]
    assert session.calls[1]["url"] == "http://mirror.local/api/v1/transactions?limit=1&timestamp=gt:1.000000001"


def test_get_next_page_rejects_missing_next_link():
    client = MirrorNodeClient(base_url="http://mirror.local", session=FakeSession([]))
    page = MirrorNodePage(items=[], next=None)

    with pytest.raises(ValueError, match="next page"):
        client.get_next_page(page, MirrorNodeTransaction.from_json)


def test_limit_validation_is_strict():
    client = MirrorNodeClient(base_url="http://mirror.local", session=FakeSession([]))

    with pytest.raises(TypeError, match="limit"):
        client.list_transactions(limit=True)

    with pytest.raises(ValueError, match="between 1 and 100"):
        client.list_transactions(limit=101)


def test_http_errors_include_status_code_and_body():
    session = FakeSession([FakeResponse(status_code=404, payload={"message": "not found"}, text="not found")])
    client = MirrorNodeClient(base_url="http://mirror.local", session=session)

    with pytest.raises(MirrorNodeError) as exc_info:
        client.get_account("0.0.404")

    assert exc_info.value.status_code == 404
    assert exc_info.value.response_body == "not found"


def test_invalid_json_and_non_object_json_raise_mirror_node_error():
    bad_json_client = MirrorNodeClient(base_url="http://mirror.local", session=FakeSession([FakeResponse(payload=ValueError())]))
    list_json_client = MirrorNodeClient(base_url="http://mirror.local", session=FakeSession([FakeResponse(payload=[])]))

    with pytest.raises(MirrorNodeError, match="valid JSON"):
        bad_json_client.get_json("/api/v1/accounts/0.0.1001")

    with pytest.raises(MirrorNodeError, match="must be an object"):
        list_json_client.get_json("/api/v1/accounts/0.0.1001")


def test_malformed_payloads_raise_clear_errors():
    with pytest.raises(MirrorNodeError, match="required string field 'account'"):
        MirrorNodeAccount.from_json({"balance": {"balance": 1}})

    with pytest.raises(MirrorNodeError, match="tokens field must be a list"):
        MirrorNodeBalance.from_json({"balance": {"balance": 1, "tokens": {}}})
