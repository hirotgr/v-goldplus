#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 設定
# =========================

DATA_DIR = Path("./")

CSV_SPXTR = DATA_DIR / "SP_SPXTR_1D.csv"
CSV_SPX   = DATA_DIR / "CBOE_DLY_SPX_1D.csv"
CSV_GOLD  = DATA_DIR / "BBG_BCOMGC_1D.csv"
CSV_FX    = DATA_DIR / "MUFG_USD_TTM.csv"
CSV_FUND  = DATA_DIR / "fund_info_645066_202601220907.csv"

BASE_DATE = "2022-08-31"   # 設定日基準（実績CSVに合わせる）
BASE_NAV  = 10000.0

DIV_TAX_RATE = 0.10        # 配当課税率（近似）
TER_ANNUAL = 0.0025        # 総経費率 0.25%/年（運用報告書の総経費率）
TRADING_DAYS = 252         # 日次控除用（投信の営業日を想定）

# 金（BCOMGC）の為替適用モード
# - full  : 金リターン（USD）に為替リターン（USD/JPY）をフルに適用（通常の外貨資産換算）
# - none  : 金リターン（USD）のみを使用（為替の影響を無視）※比較用の極端ケース
# - pnl_fx: 差金決済を簡易近似。金の損益（=リターン）にのみ為替変動を掛ける（FX単独では影響しない）
GOLD_FX_MODE = "pnl_fx"      # "full" / "none" / "pnl_fx"


# =========================
# ユーティリティ
# =========================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """TradingView等のCSVの列名揺れをある程度吸収する。"""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_tradingview_csv(path: Path, value_col_candidates=("close", "adj close", "value")) -> pd.Series:
    """
    TradingView export系CSVを読み、DatetimeIndexのSeries（終値）を返す。
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # date/datetime列名候補
    date_col = None
    for c in ("time", "date", "datetime"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"[{path.name}] date column not found in {df.columns.tolist()}")

    # value列名候補
    val_col = None
    for c in value_col_candidates:
        c2 = c.lower()
        if c2 in df.columns:
            val_col = c2
            break
    if val_col is None:
        # TradingViewのCSVで "close" を使用
        # それ以外なら、最後の列を採用
        val_col = df.columns[-1]

    s = (
        df[[date_col, val_col]]
        .rename(columns={date_col: "date", val_col: "value"})
    )
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").set_index("date")["value"].astype(float)
    return s


def load_mufg_ttm_csv(path: Path) -> pd.Series:
    """
    MUFG_USD_TTM.csv の読み込み（列名不明のため柔軟に処理）
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # date列推定
    date_col = None
    for c in ("date", "time", "datetime"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # rate列推定
    rate_col = None
    for c in ("ttm", "usd/jpy", "usdjpy", "rate", "value"):
        if c in df.columns:
            rate_col = c
            break
    if rate_col is None:
        rate_col = df.columns[1]

    s = df[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "fx"})
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").set_index("date")["fx"]
    s = pd.to_numeric(s, errors="coerce")
    return s


def _dedupe_datetime_index(s: pd.Series, label: str) -> pd.Series:
    """DatetimeIndexの重複を除去する（pandas concat/reindex対策）。

    CSV側の仕様差や二重出力等で同一日時が複数行になることがあるため、
    インデックス重複は最後の値を採用して落とす。
    """
    if s.index.has_duplicates:
        dup_n = int(s.index.duplicated(keep=False).sum())
        uniq_n = int(s.index.nunique())
        print(f"[WARN] duplicate datetime index detected in '{label}': dup_rows={dup_n}, unique_ts={uniq_n}. Keeping last occurrence.")
        s = s[~s.index.duplicated(keep="last")]
    return s


def bfill_to_calendar(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    複数Seriesを日付で結合し、欠損は翌営業日（bfill）で補完。
    """
    # concat/reindex時に "cannot reindex on an axis with duplicate labels" が出る場合は、
    # いずれかのSeriesのDatetimeIndexに重複がある。
    series_dict = {k: _dedupe_datetime_index(v.sort_index(), k) for k, v in series_dict.items()}

    df = pd.concat(series_dict, axis=1).sort_index()

    # 重要：営業日カレンダーの統一
    # データがある日付集合で回す（祝日問題を避ける）
    # ただしbfillするので「翌営業日」を模した挙動になる
    df = df.bfill()
    return df


def align_assets_with_next_day_fx(df_assets: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """米国資産データ（日付=米国市場日）に対し、翌営業日（日本側）のTTMを適用して結合する。

    - 資産データは米国市場日の終値（TradingView export）を想定。
    - 為替TTMは日本時間午前に公表されるため、米国市場日の評価には「翌営業日TTM」を適用する。
    - 休日等でTTMが欠ける場合は、次に公表されるレート（forward）を採用する。

    具体的には `pd.merge_asof(..., direction='forward')` を用いて、
    各資産日 t に対して t+1日以降で最初に存在するFX日付のレートを結合する。
    """
    if not isinstance(df_assets.index, pd.DatetimeIndex):
        raise TypeError("df_assets index must be DatetimeIndex")

    df_assets = df_assets.sort_index().copy()

    # 資産側の欠損は直前値で補完（未来値の混入を避ける）
    df_assets = df_assets.ffill()

    df_fx = fx.sort_index().rename("usdjpy").to_frame()

    tmp = df_assets.reset_index().rename(columns={df_assets.index.name or "index": "asset_date"})
    tmp["target_date"] = tmp["asset_date"] + pd.Timedelta(days=1)

    merged = pd.merge_asof(
        tmp.sort_values("target_date"),
        df_fx,
        left_on="target_date",
        right_index=True,
        direction="forward",
        tolerance=pd.Timedelta(days=10),
    )

    # FXが結合できない末尾などを除外
    merged = merged.dropna(subset=["usdjpy"]).set_index("asset_date")
    merged = merged.drop(columns=["target_date"])

    return merged


# =========================
# 配当税控除後（疑似Net TR）
# =========================

def make_tax_adjusted_tr(pr: pd.Series, tr: pd.Series, div_tax_rate: float) -> pd.Series:
    """
    PRとTRから配当成分を抽出し、配当税率を反映した疑似Net TRの日次リターン系列を返す。
    返すのは「指数水準」ではなく「日次リターン」。
    """
    r_pr = pr.pct_change()
    r_tr = tr.pct_change()

    # 配当成分（近似）
    r_div = r_tr - r_pr

    # 税引後TRリターン
    r_tr_net = r_pr + (1.0 - div_tax_rate) * r_div
    return r_tr_net


# =========================
# シミュレーション本体
# =========================

def simulate_nav(df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns:
      - spx_pr
      - spx_tr
      - bcomgc
      - usdjpy
    """
    out = df.copy()

    # 1) 株：疑似Net TRリターン（USDベース）
    out["r_eq_usd"] = make_tax_adjusted_tr(
        pr=out["spx_pr"],
        tr=out["spx_tr"],
        div_tax_rate=DIV_TAX_RATE
    )

    # 2) 金：BCOMGC（USD Excess Return）
    out["r_gold_usd"] = out["bcomgc"].pct_change()

    # 3) FX
    out["r_fx"] = out["usdjpy"].pct_change()

    # 4) 円建て株式リターン（為替フル適用）
    # (1+r_eq_jpy) = (1+r_eq_usd)*(1+r_fx)
    out["r_eq_jpy"] = (1 + out["r_eq_usd"]) * (1 + out["r_fx"]) - 1

    # 5) 円建て金リターン
    if GOLD_FX_MODE == "full":
        # 通常の外貨資産換算：金の元本も損益も為替の影響を受ける
        out["r_gold_jpy"] = (1 + out["r_gold_usd"]) * (1 + out["r_fx"]) - 1
    elif GOLD_FX_MODE == "none":
        # 比較用：金リターン（USD）のみ。為替変動を完全に無視
        out["r_gold_jpy"] = out["r_gold_usd"]
    elif GOLD_FX_MODE == "pnl_fx":
        # 差金決済の簡易近似：金の損益（リターン）にだけ為替変動を掛ける。
        # r_gold_usd=0 のとき、為替単独の変動は金部分に影響しない。
        out["r_gold_jpy"] = out["r_gold_usd"] * (1 + out["r_fx"])
    else:
        raise ValueError("Unknown GOLD_FX_MODE")

    # 6) 合成（株100% + 金100%）
    out["r_gross"] = out["r_eq_jpy"] + out["r_gold_jpy"]

    # 7) 総経費率（TER）日次控除
    ter_daily = TER_ANNUAL / TRADING_DAYS
    out["r_net"] = out["r_gross"] - ter_daily

    # 8) NAV計算（BASE_DATEで10000に正規化）
    out["nav_raw"] = np.nan

    # 初期化：BASE_DATEの次営業日から積み上げたいので、日次リターンのNaN処理
    out["r_net"] = out["r_net"].fillna(0.0)

    # nav_rawを仮に10000開始で計算
    nav = [BASE_NAV]
    idx = out.index.tolist()
    for i in range(1, len(idx)):
        nav.append(nav[-1] * (1.0 + out["r_net"].iloc[i]))
    out["nav_raw"] = nav

    # BASE_DATEが途中にある場合はそこを10000に合わせてスケール
    if BASE_DATE in out.index.strftime("%Y-%m-%d").tolist():
        # BASE_DATEのindex位置
        base_ts = pd.to_datetime(BASE_DATE)
        if base_ts in out.index:
            k = BASE_NAV / out.loc[base_ts, "nav_raw"]
            out["nav"] = out["nav_raw"] * k
        else:
            out["nav"] = out["nav_raw"]
    else:
        out["nav"] = out["nav_raw"]

    return out


# =========================
# 実績CSV読み込み＆突合
# =========================

def load_fund_nav_csv(path: Path) -> pd.Series:
    """
    fund_info_645066_xxx.csv から基準価額列を抽出
    """
    # 一部のCSVは1行目がタイトル行、2行目がヘッダになっている。
    # 1行目に日付系ヘッダの手掛かりが無い場合は1行スキップして読み直す。
    header_tokens = ("基準日", "日付", "Date", "date")

    def _first_line_has_date_header(p: Path) -> bool:
        try:
            with p.open("r", encoding="utf-8-sig", errors="replace") as f:
                first = f.readline().strip("\ufeff\n\r")
        except Exception:
            # 読めない場合は安全側：スキップしない（既存挙法）
            return True
        if not first:
            return True
        return any(tok in first for tok in header_tokens)

    skiprows = 0 if _first_line_has_date_header(path) else 1
    if skiprows == 1:
        print(f"[INFO] fund CSV seems to have a title row; skipping first line: {path.name}")

    df = pd.read_csv(path, skiprows=skiprows)
    df.columns = [c.strip() for c in df.columns]

    # 日付列推定
    date_col = None
    for c in df.columns:
        if "日付" in c or c.lower() in ("date", "datetime"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # 基準価額列推定
    nav_col = None
    for c in df.columns:
        if "基準価額" in c and "税引前" not in c:
            nav_col = c
            break
    if nav_col is None:
        # fallback
        for c in df.columns:
            if "基準価額" in c:
                nav_col = c
                break
    if nav_col is None:
        raise ValueError(f"[{path.name}] NAV column not found: {df.columns.tolist()}")

    s = df[[date_col, nav_col]].rename(columns={date_col: "date", nav_col: "nav"})
    s["date"] = pd.to_datetime(s["date"])
    s["nav"] = pd.to_numeric(s["nav"], errors="coerce")
    s = s.sort_values("date").set_index("date")["nav"]
    return s


def evaluate(sim: pd.DataFrame, fund_nav: pd.Series) -> pd.DataFrame:
    """
    シミュレーションNAVと実績NAVを突合して誤差指標を算出
    """
    df = pd.DataFrame({
        "nav_sim": sim["nav"],
        "nav_fund": fund_nav
    }).dropna()

    df["diff"] = df["nav_sim"] - df["nav_fund"]
    df["diff_pct"] = df["diff"] / df["nav_fund"]

    # 指標
    mae = df["diff"].abs().mean()
    rmse = np.sqrt((df["diff"] ** 2).mean())
    mape = df["diff_pct"].abs().mean()

    print("=== Evaluation (Sim vs Fund) ===")
    print(f"MAE  : {mae:,.2f} JPY")
    print(f"RMSE : {rmse:,.2f} JPY")
    print(f"MAPE : {mape*100:,.3f} %")
    print("===============================")

    return df


# =========================
# main
# =========================

def main():
    # Load
    spxtr = load_tradingview_csv(CSV_SPXTR)
    spx   = load_tradingview_csv(CSV_SPX)
    gold  = load_tradingview_csv(CSV_GOLD)
    fx    = load_mufg_ttm_csv(CSV_FX)

    # Merge assets on the US market calendar (avoid look-ahead). Use ffill for small gaps.
    df_assets = pd.concat({
        "spx_tr": spxtr,
        "spx_pr": spx,
        "bcomgc": gold,
    }, axis=1).sort_index()

    # Align FX: apply next-business-day (Japan) TTM to each US market day.
    df = align_assets_with_next_day_fx(df_assets, fx)

    # 最古日付（全系列揃う地点）
    df = df.dropna()

    # シミュレーション
    sim = simulate_nav(df)

    # 実績読み込み
    fund_nav = load_fund_nav_csv(CSV_FUND)

    # 評価（共通期間）
    # eval_df = evaluate(sim, fund_nav)

    # 保存
    out_csv = DATA_DIR / "sim_tracers_sp500_goldplus_nav.csv"
    sim_out = sim[["nav", "r_eq_jpy", "r_gold_jpy", "r_gross", "r_net"]].copy()
    sim_out.to_csv(out_csv, index=True, encoding="utf-8-sig")
    print(f"\nSaved simulation CSV: {out_csv}")

    # out_eval = DATA_DIR / "sim_vs_fund_eval.csv"
    # eval_df.to_csv(out_eval, index=True, encoding="utf-8-sig")
    # print(f"Saved evaluation CSV: {out_eval}")


if __name__ == "__main__":
    main()
