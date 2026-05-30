"""Typed, testable client for Hiero/Hedera mirror-node REST APIs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests

from hiero_sdk_python.account.account_id import AccountId
from hiero_sdk_python.hbar import Hbar
from hiero_sdk_python.hbar_unit import HbarUnit
from hiero_sdk_python.tokens.token_id import TokenId
from hiero_sdk_python.transaction.transaction_id import TransactionId


JsonObject = dict[str, Any]
MirrorNodeNetwork = Literal["mainnet", "testnet", "previewnet", "local-node"]


class MirrorNodeError(Exception):
    """Raised when a mirror node request fails or returns malformed data."""

    def __init__(self, message: str, *, status_code: int | None = None, response_body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass(frozen=True)
class MirrorNodeBalance:
    """Native currency and token balances returned by a mirror node."""

    hbars: Hbar
    tokens: dict[TokenId, int] = field(default_factory=dict)
    timestamp: str | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MirrorNodeBalance:
        """Build a balance from the mirror node account balance payload."""
        balance = data.get("balance", {})
        if not isinstance(balance, Mapping):
            raise MirrorNodeError("Mirror node balance payload is missing the balance object.")

        tinybars = _require_int(balance, "balance")
        tokens: dict[TokenId, int] = {}
        token_entries = balance.get("tokens", [])
        if token_entries is None:
            token_entries = []
        if not isinstance(token_entries, list):
            raise MirrorNodeError("Mirror node balance tokens field must be a list.")

        for entry in token_entries:
            if not isinstance(entry, Mapping):
                raise MirrorNodeError("Mirror node balance token entry must be an object.")
            token_id = TokenId.from_string(_require_str(entry, "token_id"))
            tokens[token_id] = _require_int(entry, "balance")

        return cls(hbars=Hbar.from_tinybars(tinybars), tokens=tokens, timestamp=_optional_str(balance, "timestamp"))


@dataclass(frozen=True)
class MirrorNodeAccount:
    """Subset of account data most callers need from `/api/v1/accounts/{id}`."""

    account_id: AccountId
    balance: MirrorNodeBalance | None = None
    alias: str | None = None
    deleted: bool | None = None
    evm_address: str | None = None
    key: JsonObject | None = None
    memo: str | None = None
    receiver_sig_required: bool | None = None
    staking_info: JsonObject | None = None
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MirrorNodeAccount:
        """Build an account model from a mirror node account payload."""
        account = _require_str(data, "account")
        key = data.get("key")
        staking_info = data.get("staking_info")
        return cls(
            account_id=AccountId.from_string(account),
            balance=MirrorNodeBalance.from_json(data) if isinstance(data.get("balance"), Mapping) else None,
            alias=_optional_str(data, "alias"),
            deleted=_optional_bool(data, "deleted"),
            evm_address=_optional_str(data, "evm_address"),
            key=dict(key) if isinstance(key, Mapping) else None,
            memo=_optional_str(data, "memo"),
            receiver_sig_required=_optional_bool(data, "receiver_sig_required"),
            staking_info=dict(staking_info) if isinstance(staking_info, Mapping) else None,
            raw=dict(data),
        )


@dataclass(frozen=True)
class MirrorNodeToken:
    """Token metadata returned by `/api/v1/tokens/{id}`."""

    token_id: TokenId
    name: str | None = None
    symbol: str | None = None
    decimals: int | None = None
    total_supply: int | None = None
    treasury_account_id: AccountId | None = None
    token_type: str | None = None
    deleted: bool | None = None
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MirrorNodeToken:
        """Build token metadata from a mirror node token payload."""
        token_id = TokenId.from_string(_require_str(data, "token_id"))
        treasury = _optional_str(data, "treasury_account_id")
        return cls(
            token_id=token_id,
            name=_optional_str(data, "name"),
            symbol=_optional_str(data, "symbol"),
            decimals=_optional_int(data, "decimals"),
            total_supply=_optional_int(data, "total_supply"),
            treasury_account_id=AccountId.from_string(treasury) if treasury is not None else None,
            token_type=_optional_str(data, "type"),
            deleted=_optional_bool(data, "deleted"),
            raw=dict(data),
        )


@dataclass(frozen=True)
class MirrorNodeTransaction:
    """Transaction summary returned by mirror node transaction endpoints."""

    transaction_id: TransactionId
    consensus_timestamp: str
    result: str | None = None
    name: str | None = None
    charged_tx_fee: Hbar | None = None
    max_fee: Hbar | None = None
    memo_base64: str | None = None
    node_account_id: AccountId | None = None
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MirrorNodeTransaction:
        """Build a transaction model from a mirror node transaction payload."""
        node_account_id = _optional_str(data, "node")
        return cls(
            transaction_id=TransactionId.from_string(_require_str(data, "transaction_id")),
            consensus_timestamp=_require_str(data, "consensus_timestamp"),
            result=_optional_str(data, "result"),
            name=_optional_str(data, "name"),
            charged_tx_fee=_optional_hbar(data, "charged_tx_fee"),
            max_fee=_optional_hbar(data, "max_fee"),
            memo_base64=_optional_str(data, "memo_base64"),
            node_account_id=AccountId.from_string(node_account_id) if node_account_id is not None else None,
            raw=dict(data),
        )


@dataclass(frozen=True)
class MirrorNodePage:
    """One mirror node collection page plus its optional next link."""

    items: list[Any]
    next: str | None
    raw: JsonObject = field(default_factory=dict)

    @property
    def has_next(self) -> bool:
        """Return True when the mirror node included a `links.next` URL."""
        return self.next is not None


class MirrorNodeClient:
    """Small synchronous mirror-node REST client with typed response helpers.

    The client deliberately wraps only common read paths and keeps raw JSON on every
    model so callers can access fields the SDK has not promoted to first-class
    attributes yet. The underlying HTTP session is injectable to make the client
    easy to verify with unit tests or custom transports.
    """

    DEFAULT_BASE_URLS: ClassVar[dict[str, str]] = {
        "mainnet": "https://mainnet-public.mirrornode.hedera.com",
        "testnet": "https://testnet.mirrornode.hedera.com",
        "previewnet": "https://previewnet.mirrornode.hedera.com",
        "local-node": "http://localhost:5551",
    }

    def __init__(
        self,
        base_url: str | None = None,
        *,
        network: MirrorNodeNetwork | str | None = "testnet",
        session: requests.Session | None = None,
        timeout: float = 10.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if base_url is None:
            if network is None:
                raise ValueError("Either base_url or network must be provided.")
            base_url = self.DEFAULT_BASE_URLS.get(str(network), str(network))
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Mirror node base_url must start with http:// or https://.")
        if timeout <= 0:
            raise ValueError("Mirror node timeout must be greater than zero.")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.headers = dict(headers or {})

    @classmethod
    def for_network(cls, network: MirrorNodeNetwork | str, **kwargs: Any) -> MirrorNodeClient:
        """Create a client for a named network or a custom base URL string."""
        return cls(network=network, **kwargs)

    def get_account(self, account_id: AccountId | str, *, transactions: bool | None = None) -> MirrorNodeAccount:
        """Fetch account metadata by account ID, alias, or EVM address."""
        params = {"transactions": str(transactions).lower()} if transactions is not None else None
        data = self.get_json(f"/api/v1/accounts/{_id_to_str(account_id)}", params=params)
        return MirrorNodeAccount.from_json(data)

    def get_account_balance(self, account_id: AccountId | str) -> MirrorNodeBalance:
        """Fetch only the native and token balances for an account."""
        return self.get_account(account_id).balance or MirrorNodeBalance(hbars=Hbar.ZERO)

    def get_token(self, token_id: TokenId | str) -> MirrorNodeToken:
        """Fetch token metadata by token ID."""
        data = self.get_json(f"/api/v1/tokens/{_id_to_str(token_id)}")
        return MirrorNodeToken.from_json(data)

    def get_transaction(self, transaction_id: TransactionId | str) -> MirrorNodeTransaction:
        """Fetch the first transaction matching a transaction ID."""
        page = self.list_transactions(transaction_id=transaction_id, limit=1)
        if not page.items:
            raise MirrorNodeError(f"No transaction found for transaction_id {transaction_id}.")
        return page.items[0]

    def list_transactions(
        self,
        *,
        account_id: AccountId | str | None = None,
        transaction_id: TransactionId | str | None = None,
        result: str | None = None,
        transaction_type: str | None = None,
        order: Literal["asc", "desc"] | None = None,
        limit: int | None = None,
    ) -> MirrorNodePage:
        """List transactions with common mirror node filters."""
        params: dict[str, Any] = {}
        if account_id is not None:
            params["account.id"] = _id_to_str(account_id)
        if transaction_id is not None:
            params["transactionid"] = str(transaction_id)
        if result is not None:
            params["result"] = result
        if transaction_type is not None:
            params["type"] = transaction_type
        if order is not None:
            params["order"] = order
        if limit is not None:
            _validate_limit(limit)
            params["limit"] = limit

        data = self.get_json("/api/v1/transactions", params=params)
        return self._page(data, "transactions", MirrorNodeTransaction.from_json)

    def iterate_transactions(self, *, max_pages: int | None = None, **filters: Any) -> Iterator[MirrorNodeTransaction]:
        """Yield transactions across pages until the result set is exhausted."""
        page_count = 0
        page = self.list_transactions(**filters)
        while True:
            page_count += 1
            yield from page.items
            if page.next is None or (max_pages is not None and page_count >= max_pages):
                return
            page = self.get_next_page(page, MirrorNodeTransaction.from_json)

    def get_next_page(self, page: MirrorNodePage, item_factory: Any) -> MirrorNodePage:
        """Fetch a previously returned `links.next` page using the supplied item factory."""
        if page.next is None:
            raise ValueError("Cannot fetch a next page when page.next is None.")
        data = self.get_json(page.next)
        collection_name = _detect_collection_name(data)
        return self._page(data, collection_name, item_factory)

    def get_json(self, path_or_url: str, *, params: Mapping[str, Any] | None = None) -> JsonObject:
        """Perform a GET request and return a JSON object, raising MirrorNodeError on failure."""
        url = self._build_url(path_or_url, params=params)
        response = self.session.get(url, timeout=self.timeout, headers=self.headers)
        if response.status_code >= 400:
            raise MirrorNodeError(
                f"Mirror node request failed with HTTP {response.status_code}.",
                status_code=response.status_code,
                response_body=response.text,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise MirrorNodeError("Mirror node response did not contain valid JSON.", response_body=response.text) from exc
        if not isinstance(data, dict):
            raise MirrorNodeError("Mirror node response JSON must be an object.")
        return data

    def _build_url(self, path_or_url: str, *, params: Mapping[str, Any] | None = None) -> str:
        parsed = urlparse(path_or_url)
        url = path_or_url if parsed.scheme and parsed.netloc else urljoin(f"{self.base_url}/", path_or_url.lstrip("/"))
        return _merge_query(url, params)

    @staticmethod
    def _page(data: Mapping[str, Any], collection_name: str, item_factory: Any) -> MirrorNodePage:
        collection = data.get(collection_name)
        if not isinstance(collection, list):
            raise MirrorNodeError(f"Mirror node response is missing the {collection_name} list.")
        links = data.get("links", {})
        next_link = links.get("next") if isinstance(links, Mapping) else None
        if next_link is not None and not isinstance(next_link, str):
            raise MirrorNodeError("Mirror node links.next value must be a string or null.")
        return MirrorNodePage(items=[item_factory(item) for item in collection], next=next_link, raw=dict(data))


def _id_to_str(value: AccountId | TokenId | str) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("Identifier strings must not be empty.")
        return value
    return str(value)


def _merge_query(url: str, params: Mapping[str, Any] | None) -> str:
    if not params:
        return url
    parsed = urlparse(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            query_items.extend((key, str(item)) for item in value)
        else:
            query_items.append((key, str(value)))
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _detect_collection_name(data: Mapping[str, Any]) -> str:
    collection_names = [key for key, value in data.items() if isinstance(value, list) and key != "links"]
    if len(collection_names) != 1:
        raise MirrorNodeError("Unable to detect collection name in mirror node page.")
    return collection_names[0]


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise MirrorNodeError(f"Mirror node payload is missing required string field '{key}'.")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MirrorNodeError(f"Mirror node field '{key}' must be a string when present.")
    return value


def _require_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise MirrorNodeError(f"Mirror node field '{key}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise MirrorNodeError(f"Mirror node payload is missing required integer field '{key}'.")


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _require_int(data, key)


def _optional_bool(data: Mapping[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise MirrorNodeError(f"Mirror node field '{key}' must be a boolean when present.")
    return value


def _optional_hbar(data: Mapping[str, Any], key: str) -> Hbar | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MirrorNodeError(f"Mirror node field '{key}' must be an integer tinybar amount when present.")
    return Hbar(value, HbarUnit.TINYBAR)
