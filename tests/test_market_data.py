import unittest
from unittest.mock import patch

import pandas as pd

from market_data import get_hourly_candles


def _fake_download_response():
    index = pd.date_range("2026-08-25 10:00", periods=3, freq="60min")
    columns = pd.MultiIndex.from_product([
        ["Open", "High", "Low", "Close", "Volume"],
        ["PETR4.SA"],
    ])
    return pd.DataFrame(
        [
            [37.10, 37.35, 37.05, 37.30, 1_200_000],
            [37.30, 37.50, 37.20, 37.45, 980_000],
            [37.45, 37.60, 37.40, 37.55, 1_050_000],
        ],
        index=index,
        columns=columns,
    )


class GetHourlyCandlesTests(unittest.TestCase):
    @patch("market_data.yf.download")
    def test_returns_flattened_ohlcv_dataframe(self, mock_download):
        mock_download.return_value = _fake_download_response()

        result = get_hourly_candles(ticker="PETR4.SA", period="5d", interval="60m")

        mock_download.assert_called_once_with(
            "PETR4.SA", period="5d", interval="60m", progress=False
        )
        self.assertListEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result.iloc[0]["Close"], 37.30)

    @patch("market_data.yf.download")
    def test_raises_on_empty_response(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            get_hourly_candles(ticker="INVALID.SA")


if __name__ == "__main__":
    unittest.main()
