"""Memuat model tersimpan dan menghasilkan ramalan harga komoditas (tanpa tuning ulang)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.iolib.smpickle import load_pickle

ROOT = Path(__file__).resolve().parents[1]
DIR_MODEL = ROOT / "models" / "forecasting"
DIR_DATA = ROOT / "data" / "processed" / "forecasting"


def muat_data():
    """Baca deret harga dan fitur kalender dari berkas hasil tahap persiapan."""
    deret = pd.read_csv(DIR_DATA / "deret_waktu_harian.csv", parse_dates=["tanggal"])
    kal = pd.read_csv(DIR_DATA / "kalender_fitur.csv", parse_dates=["tanggal"]).set_index("tanggal")
    harga = {n: g.sort_values("tanggal").set_index("tanggal")["harga"].astype(float)
             for n, g in deret.groupby("komoditas")}
    return harga, kal.select_dtypes(include=["number", "bool"]).astype(float)


def muat_metadata():
    return json.loads((DIR_MODEL / "metadata_model.json").read_text(encoding="utf-8"))


def _validasi(nama, y, kal, cfg, h):
    """Pastikan deret kontinu, cukup panjang, dan tidak lebih pendek dari data latih model."""
    if y.empty or y.isna().any() or (y <= 0).any():
        raise ValueError(f"{nama}: ada harga kosong atau tidak positif")
    if not (y.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError(f"{nama}: deret harga tidak kontinu")
    if cfg["model"] != "naive" and y.index[-1] < pd.Timestamp(cfg["data_akhir"]):
        raise ValueError(f"{nama}: data berakhir {y.index[-1].date()}, lebih awal dari data latih model "
                         f"({cfg['data_akhir']})")
    if cfg["model"] == "xgboost":
        if len(y) < 29:
            raise ValueError(f"{nama}: data terlalu pendek untuk fitur XGBoost (minimum 29 hari)")
        if kal.index.max() < y.index[-1] + pd.Timedelta(days=h):
            raise ValueError(f"{nama}: kalender hanya sampai {kal.index.max().date()}, "
                             f"perlu sampai {(y.index[-1] + pd.Timedelta(days=h)).date()}")


def _fitur_xgb(y, kal, lags, horizon):
    """Satu baris per (tanggal asal t, h). Fitur hanya memakai data <= t dan kalender tanggal t+h."""
    F = pd.DataFrame({f"sel_{k}": y - y.shift(k) for k in lags})
    F["dev_m7"], F["dev_m28"] = y - y.rolling(7).mean(), y - y.rolling(28).mean()
    F["std_7"] = y.rolling(7).std()
    return pd.concat([F.assign(h=h, tgl_target=F.index + pd.Timedelta(days=h)).join(kal, on="tgl_target")
                      for h in range(1, horizon + 1)])


def ramal(nama, y, kal, meta):
    """Ramalan HORIZON hari setelah data terakhir untuk satu komoditas."""
    h, cfg = meta["horizon"], meta["komoditas"][nama]
    _validasi(nama, y, kal, cfg, h)

    if cfg["model"] == "naive":
        pred = np.repeat(y.iloc[-1], h)

    elif cfg["model"] in ("arima", "sarima"):
        hasil = load_pickle(DIR_MODEL / cfg["berkas_model"])
        baru = y.loc[y.index > pd.Timestamp(cfg["data_akhir"])].asfreq("D")
        if len(baru):
            hasil = hasil.append(baru, refit=False)         # perbarui state, parameter tetap terkunci
        pred = np.asarray(hasil.forecast(h))

    elif cfg["model"] == "xgboost":
        booster = xgb.Booster()
        booster.load_model(DIR_MODEL / cfg["berkas_model"])
        D = _fitur_xgb(y, kal, meta["lags_xgboost"], h)
        asal = D.loc[y.index[-1]].sort_values("h")
        pred = booster.predict(xgb.DMatrix(asal[cfg["fitur_xgboost"]])) + y.iloc[-1]

    else:
        raise ValueError(f"{nama}: model '{cfg['model']}' tidak dikenal")

    tgl = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)
    return pd.DataFrame({"komoditas": nama, "model": cfg["model"], "tanggal": tgl,
                         "tanggal_asal": y.index[-1], "h": np.arange(1, h + 1),
                         "ramalan": np.round(pred)})


def ramal_semua(harga=None, kal=None):
    """Ramalan semua komoditas. Bila harga dan kal tidak diberikan, data dibaca dari berkas CSV.
    harga: dict {nama komoditas: pd.Series harga harian berindeks tanggal}."""
    if harga is None or kal is None:
        harga, kal = muat_data()
    meta = muat_metadata()
    return pd.concat([ramal(n, y, kal, meta) for n, y in harga.items()], ignore_index=True)