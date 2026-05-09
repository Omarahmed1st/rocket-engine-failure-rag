import pandas as pd

def generate_summary(df, file_name):

    avg_thrust = df["thrust"].mean()

    max_thrust = df["thrust"].max()

    avg_mfr = df["mfr"].mean()

    avg_vl = df["vl"].mean()

    # extract anomaly codes
    anomalies = (
        df["anomaly_code"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    # ==========================================
    # THRUST DESCRIPTION
    # ==========================================

    if avg_thrust > 1:

        thrust_desc = "high thrust behavior"

    elif avg_thrust > 0.5:

        thrust_desc = "moderate thrust behavior"

    else:

        thrust_desc = "low thrust behavior"

    # ==========================================
    # VALVE DESCRIPTION
    # ==========================================

    if avg_vl > 0.5:

        valve_desc = "unstable valve activity"

    else:

        valve_desc = "stable valve activity"

    # ==========================================
    # ANOMALY DESCRIPTION
    # ==========================================

    if len(anomalies) > 0:

        anomaly_text = (
            f"Detected anomaly codes: {anomalies}."
        )

    else:

        anomaly_text = (
            "No anomalies were detected."
        )

    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    summary = f"""
    Thruster firing test analysis.

    File name: {file_name}

    The test demonstrated {thrust_desc}.

    Average thrust level was {avg_thrust:.4f}.

    Maximum thrust reached {max_thrust:.4f}.

    Average mass flow rate was {avg_mfr:.4f}.

    The propulsion system showed {valve_desc}.

    Average valve level was {avg_vl:.4f}.

    {anomaly_text}

    The thruster firing sequence showed propulsion activity,
    thrust variation, and valve response changes.
    """

    return summary