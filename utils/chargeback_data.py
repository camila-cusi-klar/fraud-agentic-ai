from __future__ import annotations

from typing import Any

import pandas as pd

from utils.data_ingest import get_db_conn


CB_QUERY = """
WITH tc_base AS (
    SELECT
        tc.*,
        COALESCE(
            NULLIF(split_part(tc.transaction_id, 'PARABILIUM:', 2), ''),
            tc.transaction_id
        ) AS omi_id_operacion_raw
    FROM ops_fraud.total_chargeback tc
),

oimt_base AS (
    SELECT
        oimt.omi_id_operacion,
        oimt.c063,
        oimt.c032 as adquirente,
        o.operador,
        o.cod_respuesta
    FROM is_pii_parabilium.operation_iso_messages_temp oimt
    INNER JOIN tc_base tc
        ON oimt.omi_id_operacion = tc.omi_id_operacion_raw
    inner join is_pii_parabilium.operaciones o
        on o.id_operacion = tc.omi_id_operacion_raw
),

pin_verif AS (
    SELECT
        o.omi_id_operacion,
        /* ---------- CVM RESULTS (Byte 1) ---------- */
        '! ' || REGEXP_SUBSTR(o.c063, 'B300080[^!]*') AS B300080_value,
        SUBSTRING(B300080_value FROM 49 FOR 6) AS "8-CVMRSLTS",
        SUBSTRING("8-CVMRSLTS" FROM 1 FOR 2) AS byte_1_hex,
        CASE
            WHEN byte_1_hex ~ '^[0-9A-Fa-f]{2}$'
            THEN FROM_VARBYTE(from_hex(byte_1_hex), 'binary')
            ELSE NULL
        END AS byte_1_bits,
        CASE
            WHEN byte_1_bits IS NULL THEN NULL
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000000'
                THEN 'Procesamiento de CVM fallido'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000001'
                THEN 'Verificacion de PIN en texto plano realizada por el chip de la tarjeta'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000010'
                THEN 'PIN cifrado verificado en linea'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000011'
                THEN 'Verificacion de PIN en texto plano realizada por el chip de la tarjeta y firma (papel)'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000100'
                THEN 'Verificacion de PIN cifrado realizada por el chip de la tarjeta'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '000101'
                THEN 'Verificacion de PIN cifrado realizada por el chip de la tarjeta y firma (papel)'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '011110'
                THEN 'Firma (papel)'
            WHEN SUBSTRING(byte_1_bits FROM 3 FOR 6) = '011111'
                THEN 'No se requiere CVM'
            ELSE SUBSTRING(byte_1_bits FROM 3 FOR 6)
        END AS metodo_verificacion
    FROM oimt_base o
    WHERE o.c063 LIKE '%! B3%'
),

three_ds AS (
    SELECT
        o.omi_id_operacion,
        o.c063,

        /* ---------- CE TOKEN EXTRACTION ---------- */
        REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*') AS ce_token,

        /* ---------- LEADING INDICATOR (kA, kG, kN, etc.) ---------- */
        CASE
            WHEN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*') IS NOT NULL
                 AND position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) > 0
            THEN substring(
                REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                FOR 2
            )
            ELSE NULL
        END AS leading_indicator,

        /* ---------- 3DS STATUS ---------- */
        CASE
            WHEN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*') IS NULL
                THEN 'NO_3DS'
            WHEN substring(
                     REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                     FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                     FOR 2
                 ) IN ('kA','kB','kC','kE','kF','kJ','kR','kS','kG','kO','kP')
                THEN '3DS_AUTHENTICATED'
            WHEN substring(
                     REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                     FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                     FOR 2
                 ) IN ('kN','kW','kU','kX')
                THEN '3DS_NOT_AUTHENTICATED'
            ELSE 'UNKNOWN'
        END AS three_ds_status,

        /* ---------- 3DS FLOW ---------- */
        CASE
            WHEN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*') IS NULL
                THEN 'NO_3DS'
            WHEN substring(
                     REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                     FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                     FOR 2
                 ) IN ('kA','kC','kE','kO')
                THEN 'FRICTIONLESS'
            WHEN substring(
                     REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                     FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                     FOR 2
                 ) IN ('kB','kS','kG','kP')
                THEN 'CHALLENGE'
            WHEN substring(
                     REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')
                     FROM position('01' IN REGEXP_SUBSTR(o.c063, 'CE[0-9]{5}[^!]*')) + 2
                     FOR 2
                 ) IN ('kN','kW','kU','kX')
                THEN 'EXEMPT_OR_INFO'
            ELSE 'UNKNOWN'
        END AS three_ds_flow
    FROM oimt_base o
)

SELECT
    tc.*,
    pv.metodo_verificacion AS metodo_identificacion,
    td.leading_indicator,
    td.three_ds_status,
    td.three_ds_flow,
    td.c063,
    o.sucursal as afiliacion,
    o.terminal as numero_terminal,
    ob.adquirente,
    o.operador,
    o.cod_respuesta
FROM tc_base tc
LEFT JOIN pin_verif pv
    ON pv.omi_id_operacion = tc.omi_id_operacion_raw
LEFT JOIN three_ds td
    ON td.omi_id_operacion = tc.omi_id_operacion_raw
left join
    is_pii_parabilium.parabilium_transactions pt on pt.id = tc.omi_id_operacion_raw
left join
    is_pii_parabilium.operaciones o on o.id_operacion = tc.omi_id_operacion_raw
LEFT JOIN oimt_base ob
    ON ob.omi_id_operacion = tc.omi_id_operacion_raw
"""


CB_REASON_MAP = {
    "Transacción por internet o compra telefónica": "Rembolso no recibido",
    "Error durante el proceso": "Cargo no reconocido",
    "Otro": "Otro",
    "Producto no recibido": "Rembolso no recibido",
    "Transacción duplicada": "Transacción duplicada",
    "Transacción declinada": "Transacción declinada",
    "Transacción excede el importe autorizado": "Rembolso no procesado",
    "Aclaración de Transacción de Cajero Automático": "Aclaración de Transacción de Cajero Automático",
    "ATM no entrega dinero": "Aclaración de Transacción de Cajero Automático",
    "Reporte de cargo no reconocido": "Cargo no reconocido",
    "TRANSACTION_EXCEEDS_AUTHORIZED_AMOUNT": "Rembolso no procesado",
    "Reembolso no procesado": "Rembolso no procesado",
    "REFUND_NOT_PROCESSED": "Rembolso no procesado",
    "TRANSACTION_DECLINED": "Transacción declinada",
    "OTHER": "Otro",
    "UNRECOGNIZED_TRANSACTION": "Cargo no reconocido",
    "INTERNET_PHONE_TRANSACTION": "Rembolso no recibido",
    "UNRECOGNIZED_CHARGE": "Cargo no reconocido",
    "DUPLICATE_TRANSACTION": "Transacción duplicada",
    "ATM_TRANSACTION_CLARIFICATION": "Aclaración de Transacción de Cajero Automático",
}


ONLINE_POS_ENTRY_MODES = {"CNP Manual", "CNP Card on File"}


def download_cb_df(
    conn: Any | None = None,
    creds_file: str = '',
    query: str = CB_QUERY,
) -> pd.DataFrame:
    """Download the raw chargeback dataframe used by the CB analysis notebooks."""
    should_close_conn = conn is None
    if conn is None:
        conn = get_db_conn(creds_file)

    try:
        return pd.read_sql(query, conn)
    finally:
        if should_close_conn:
            conn.close()


def transform_cb_df(cb_df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Apply the notebook transformations through cb_df['country']."""
    if copy:
        cb_df = cb_df.copy()

    cb_df["trx_time"] = cb_df["trx_timestamp_mx"]
    cb_df["trx_timestamp_mx"] = pd.to_datetime(cb_df["trx_timestamp_mx"])
    cb_df["week_number"] = cb_df["trx_timestamp_mx"].dt.isocalendar().week
    cb_df["trx_date"] = cb_df["trx_timestamp_mx"].dt.date
    cb_df["trx_month"] = cb_df["trx_timestamp_mx"].dt.month
    cb_df["trx_day"] = cb_df["trx_timestamp_mx"].dt.day
    cb_df["trx_month_year"] = cb_df["trx_timestamp_mx"].dt.to_period("M")
    cb_df["trx_week_year"] = cb_df["trx_timestamp_mx"].dt.strftime("%Y-%W")

    cb_df["cb_timestamp"] = pd.to_datetime(cb_df["cb_timestamp"])
    cb_df["cb_date"] = cb_df["cb_timestamp"].dt.date
    cb_df["cb_month"] = cb_df["cb_timestamp"].dt.month
    cb_df["cb_day"] = cb_df["cb_timestamp"].dt.day
    cb_df["cb_month_year"] = cb_df["cb_timestamp"].dt.to_period("M")
    cb_df["cb_week_year"] = cb_df["cb_timestamp"].dt.strftime("%Y-%W")

    cb_df = cb_df.drop_duplicates(subset=["transaction_id"], keep="first")
    cb_df = cb_df[
        (cb_df["transaction_id"].notna())
        & (cb_df["amount"].notna())
        & (cb_df["cb_reason"] != "Devolución de dinero")
    ].copy()

    cb_df["online"] = cb_df["pos_entry_mode"].isin(ONLINE_POS_ENTRY_MODES)
    cb_df["cb_reason_original"] = cb_df["cb_reason"]
    cb_df["cb_reason"] = cb_df["cb_reason"].map(CB_REASON_MAP)
    cb_df["country"] = cb_df["operador"].str[-2:]
    cb_df["amount_pos"] = cb_df["amount"]*-1
    return cb_df


def load_cb_df(
    conn: Any | None = None,
    creds_file: str  = '',
    query: str = CB_QUERY,
) -> pd.DataFrame:
    """Download and transform cb_df in one call."""
    cb_df = download_cb_df(conn=conn, creds_file=creds_file, query=query)
    return transform_cb_df(cb_df, copy=False)
