import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger("backtesting.cost_model")

class IndianTransactionCostModel:
    """
    Models realistic Indian stock and F&O market transaction costs & friction:
    - Securities Transaction Tax (STT): 0.025% sell-side (delivery), 0.1% intraday / futures
    - Exchange Transaction Fee: NSE 0.00297%, BSE 0.00375%
    - GST: 18% on (brokerage + exchange transaction fee)
    - SEBI turnover fee: 0.0001%
    - Stamp Duty: 0.003% buy side
    - Brokerage: Flat Rs 20 per executed order
    - Slippage: 1 to 5 bps (default 2 bps = 0.0002)
    """

    def __init__(
        self,
        stt_rate: float = 0.001,           # 0.1% on turnover / futures
        exchange_txn_fee: float = 0.0000297, # 0.00297%
        sebi_fee: float = 0.000001,         # 0.0001%
        stamp_duty: float = 0.00003,        # 0.003% buy side
        gst_rate: float = 0.18,             # 18% GST
        flat_brokerage: float = 20.0,       # Rs 20 per order
        slippage_bps: float = 2.0           # 2 bps slippage
    ):
        self.stt_rate = stt_rate
        self.exchange_txn_fee = exchange_txn_fee
        self.sebi_fee = sebi_fee
        self.stamp_duty = stamp_duty
        self.gst_rate = gst_rate
        self.flat_brokerage = flat_brokerage
        self.slippage = slippage_bps / 10000.0

    def calculate_round_turn_cost(self, trade_value: float, is_buy: bool = True) -> Dict[str, float]:
        """
        Calculates detailed transaction costs for a trade of given monetary value.
        """
        stt = trade_value * self.stt_rate if not is_buy else 0.0
        stamp = trade_value * self.stamp_duty if is_buy else 0.0
        exchange_fee = trade_value * self.exchange_txn_fee
        sebi_charge = trade_value * self.sebi_fee
        brokerage = self.flat_brokerage
        gst = (brokerage + exchange_fee) * self.gst_rate
        slippage_cost = trade_value * self.slippage

        total_cost = stt + stamp + exchange_fee + sebi_charge + brokerage + gst + slippage_cost
        cost_ratio = total_cost / trade_value if trade_value > 0 else 0.0

        return {
            "stt": stt,
            "stamp_duty": stamp,
            "exchange_fee": exchange_fee,
            "sebi_fee": sebi_charge,
            "brokerage": brokerage,
            "gst": gst,
            "slippage": slippage_cost,
            "total_cost": total_cost,
            "cost_bps": cost_ratio * 10000.0
        }
