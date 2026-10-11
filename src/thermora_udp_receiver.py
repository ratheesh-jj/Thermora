import socket
import json
import csv
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

BASE_DIR = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
)

OUTPUT_DIR = BASE_DIR / "results" / "live_sensor_data"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SESSION FILES
# ============================================================

session = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

imu_file = OUTPUT_DIR / f"imu_{session}.csv"
ppg_file = OUTPUT_DIR / f"ppg_{session}.csv"
vital_file = OUTPUT_DIR / f"vitals_{session}.csv"
env_file = OUTPUT_DIR / f"environment_{session}.csv"


# ============================================================
# CSV FIELDS
# ============================================================

imu_fields = [
    "timestamp_received",
    "time_ms",
    "device",
    "sender_ip",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]


ppg_fields = [
    "timestamp_received",
    "time_ms",
    "device",
    "sender_ip",
    "ir",
    "red",
]


vital_fields = [
    "timestamp_received",
    "time_ms",
    "device",
    "sender_ip",
    "hr_bpm",
    "hr_valid",
    "spo2_pct",
    "spo2_valid",
]


env_fields = [
    "timestamp_received",
    "time_ms",
    "device",
    "sender_ip",
    "ambient_temp_c",
    "humidity_pct",
]


# ============================================================
# CREATE CSV
# ============================================================

def create_csv(path, fields):

    file_exists = path.exists()

    f = open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    if not file_exists:
        writer.writeheader()
        f.flush()

    return f, writer


imu_fp, imu_writer = create_csv(
    imu_file,
    imu_fields
)

ppg_fp, ppg_writer = create_csv(
    ppg_file,
    ppg_fields
)

vital_fp, vital_writer = create_csv(
    vital_file,
    vital_fields
)

env_fp, env_writer = create_csv(
    env_file,
    env_fields
)


# ============================================================
# UDP SOCKET
# ============================================================

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

sock.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)


try:

    sock.bind(
        (UDP_IP, UDP_PORT)
    )

except OSError as e:

    print()
    print("=" * 65)
    print("ERROR: COULD NOT OPEN UDP PORT")
    print("=" * 65)

    print(e)

    print()
    print("Another program may already be using port 5005.")
    print()
    print("Run this in Command Prompt:")
    print()
    print("netstat -ano | findstr :5005")
    print()
    print("Then terminate the process using:")
    print()
    print("taskkill /PID <PID> /F")
    print()

    imu_fp.close()
    ppg_fp.close()
    vital_fp.close()
    env_fp.close()

    raise SystemExit


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 65)
print("             THERMORA LIVE SENSOR RECEIVER")
print("=" * 65)

print()
print(f"Listening on UDP port: {UDP_PORT}")

print()
print("Saving data to:")
print(OUTPUT_DIR)

print()
print(f"IMU:         {imu_file.name}")
print(f"PPG:         {ppg_file.name}")
print(f"Vitals:      {vital_file.name}")
print(f"Environment: {env_file.name}")

print()
print("Waiting for ESP32 packets...")
print("Press CTRL+C to stop.")

print("=" * 65)
print()


# ============================================================
# COUNTERS
# ============================================================

imu_count = 0
ppg_count = 0
vital_count = 0
env_count = 0
unknown_count = 0


# ============================================================
# RECEIVE LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # RECEIVE UDP PACKET
        # ----------------------------------------------------

        data, address = sock.recvfrom(8192)

        sender_ip = address[0]

        receive_time = datetime.now().isoformat(
            timespec="milliseconds"
        )


        # ----------------------------------------------------
        # DECODE JSON
        # ----------------------------------------------------

        try:

            packet = json.loads(
                data.decode("utf-8")
            )

        except Exception as e:

            print()
            print(
                f"[ERROR] Invalid JSON from "
                f"{sender_ip}: {e}"
            )

            continue


        # ----------------------------------------------------
        # BASIC PACKET INFORMATION
        # ----------------------------------------------------

        packet_type = packet.get("type")

        device = packet.get("device")

        time_ms = packet.get(
            "timestamp_ms"
        )


        # ====================================================
        # WRISTBAND PACKET
        # ====================================================

        if packet_type == "wristband":

            # ------------------------------------------------
            # GET SENSOR DATA
            # ------------------------------------------------

            imu = packet.get(
                "imu",
                {}
            )

            ppg = packet.get(
                "ppg",
                {}
            )

            vitals = packet.get(
                "vitals",
                {}
            )

            sensor_status = packet.get(
                "sensor_status",
                {}
            )


            # ------------------------------------------------
            # SENSOR STATUS
            # ------------------------------------------------

            mpu_ok = sensor_status.get(
                "mpu6050",
                False
            )

            max_ok = sensor_status.get(
                "max30102",
                False
            )


            # =================================================
            # SAVE IMU DATA
            # =================================================

            if mpu_ok:

                imu_row = {

                    "timestamp_received":
                        receive_time,

                    "time_ms":
                        time_ms,

                    "device":
                        device,

                    "sender_ip":
                        sender_ip,

                    "ax":
                        imu.get("ax"),

                    "ay":
                        imu.get("ay"),

                    "az":
                        imu.get("az"),

                    "gx":
                        imu.get("gx"),

                    "gy":
                        imu.get("gy"),

                    "gz":
                        imu.get("gz"),
                }

                imu_writer.writerow(
                    imu_row
                )

                imu_fp.flush()

                imu_count += 1


            # =================================================
            # SAVE PPG DATA
            # =================================================

            if max_ok:

                ppg_row = {

                    "timestamp_received":
                        receive_time,

                    "time_ms":
                        time_ms,

                    "device":
                        device,

                    "sender_ip":
                        sender_ip,

                    "ir":
                        ppg.get("ir"),

                    "red":
                        ppg.get("red"),
                }

                ppg_writer.writerow(
                    ppg_row
                )

                ppg_fp.flush()

                ppg_count += 1


            # =================================================
            # SAVE VITAL DATA
            # =================================================

            if max_ok:

                vital_row = {

                    "timestamp_received":
                        receive_time,

                    "time_ms":
                        time_ms,

                    "device":
                        device,

                    "sender_ip":
                        sender_ip,

                    "hr_bpm":
                        vitals.get(
                            "hr_bpm"
                        ),

                    "hr_valid":
                        vitals.get(
                            "hr_valid"
                        ),

                    "spo2_pct":
                        vitals.get(
                            "spo2_pct"
                        ),

                    "spo2_valid":
                        vitals.get(
                            "spo2_valid"
                        ),
                }

                vital_writer.writerow(
                    vital_row
                )

                vital_fp.flush()

                vital_count += 1


            # =================================================
            # TERMINAL DISPLAY
            # =================================================

            print(
                f"[WRISTBAND] {sender_ip} | "
                f"MPU={'OK' if mpu_ok else 'N/A'} | "
                f"MAX30102={'OK' if max_ok else 'N/A'}"
            )


            # ------------------------------------------------
            # HR + SPO2
            # ------------------------------------------------

            hr = vitals.get(
                "hr_bpm"
            )

            hr_valid = vitals.get(
                "hr_valid",
                False
            )

            spo2 = vitals.get(
                "spo2_pct"
            )

            spo2_valid = vitals.get(
                "spo2_valid",
                False
            )


            hr_display = (
                str(hr)
                if hr_valid and hr is not None
                else "N/A"
            )

            spo2_display = (
                str(spo2)
                if spo2_valid and spo2 is not None
                else "N/A"
            )


            print(
                f"           "
                f"HR={hr_display} BPM | "
                f"SpO2={spo2_display}%"
            )


            # ------------------------------------------------
            # RAW PPG
            # ------------------------------------------------

            if max_ok:

                print(
                    f"           "
                    f"IR={ppg.get('ir')} | "
                    f"RED={ppg.get('red')}"
                )


            # ------------------------------------------------
            # IMU
            # ------------------------------------------------

            if mpu_ok:

                print(
                    f"           "
                    f"ACC=("
                    f"{imu.get('ax', 0):.2f}, "
                    f"{imu.get('ay', 0):.2f}, "
                    f"{imu.get('az', 0):.2f}) | "
                    f"GYRO=("
                    f"{imu.get('gx', 0):.2f}, "
                    f"{imu.get('gy', 0):.2f}, "
                    f"{imu.get('gz', 0):.2f})"
                )


            print()


        # ====================================================
        # SEPARATE IMU PACKET
        # ====================================================

        elif packet_type == "IMU":

            imu_row = {

                "timestamp_received":
                    receive_time,

                "time_ms":
                    packet.get(
                        "time_ms"
                    ),

                "device":
                    device,

                "sender_ip":
                    sender_ip,

                "ax":
                    packet.get("ax"),

                "ay":
                    packet.get("ay"),

                "az":
                    packet.get("az"),

                "gx":
                    packet.get("gx"),

                "gy":
                    packet.get("gy"),

                "gz":
                    packet.get("gz"),
            }

            imu_writer.writerow(
                imu_row
            )

            imu_fp.flush()

            imu_count += 1

            print(
                f"[IMU] {sender_ip} | "
                f"ACC=("
                f"{packet.get('ax')}, "
                f"{packet.get('ay')}, "
                f"{packet.get('az')}) | "
                f"GYRO=("
                f"{packet.get('gx')}, "
                f"{packet.get('gy')}, "
                f"{packet.get('gz')})"
            )


        # ====================================================
        # SEPARATE PPG PACKET
        # ====================================================

        elif packet_type == "PPG":

            ppg_row = {

                "timestamp_received":
                    receive_time,

                "time_ms":
                    packet.get(
                        "time_ms"
                    ),

                "device":
                    device,

                "sender_ip":
                    sender_ip,

                "ir":
                    packet.get("ir"),

                "red":
                    packet.get("red"),
            }

            ppg_writer.writerow(
                ppg_row
            )

            ppg_fp.flush()

            ppg_count += 1

            print(
                f"[PPG] {sender_ip} | "
                f"IR={packet.get('ir')} | "
                f"RED={packet.get('red')}"
            )


        # ====================================================
        # SEPARATE VITAL PACKET
        # ====================================================

        elif packet_type == "VITAL":

            vital_row = {

                "timestamp_received":
                    receive_time,

                "time_ms":
                    packet.get(
                        "time_ms"
                    ),

                "device":
                    device,

                "sender_ip":
                    sender_ip,

                "hr_bpm":
                    packet.get(
                        "hr_bpm"
                    ),

                "hr_valid":
                    packet.get(
                        "hr_valid"
                    ),

                "spo2_pct":
                    packet.get(
                        "spo2_pct"
                    ),

                "spo2_valid":
                    packet.get(
                        "spo2_valid"
                    ),
            }

            vital_writer.writerow(
                vital_row
            )

            vital_fp.flush()

            vital_count += 1

            print(
                f"[VITAL] {sender_ip} | "
                f"HR={packet.get('hr_bpm')} BPM | "
                f"SpO2={packet.get('spo2_pct')}%"
            )


        # ====================================================
        # ENVIRONMENT
        # ====================================================

        elif packet_type == "ENV":

            ambient_temp = packet.get(
                "ambient_temp_c"
            )

            humidity = packet.get(
                "humidity_pct"
            )


            env_row = {

                "timestamp_received":
                    receive_time,

                "time_ms":
                    packet.get(
                        "time_ms"
                    ),

                "device":
                    device,

                "sender_ip":
                    sender_ip,

                "ambient_temp_c":
                    ambient_temp,

                "humidity_pct":
                    humidity,
            }

            env_writer.writerow(
                env_row
            )

            env_fp.flush()

            env_count += 1


            print(
                f"[ENV] {sender_ip} | "
                f"T={ambient_temp} °C | "
                f"RH={humidity}%"
            )


        # ====================================================
        # UNKNOWN PACKET
        # ====================================================

        else:

            unknown_count += 1

            print()
            print(
                f"[UNKNOWN PACKET] "
                f"from {sender_ip}"
            )

            print(
                f"Type: {packet_type}"
            )

            print(
                f"Data: {packet}"
            )

            print()


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print()
    print("Stopping receiver...")


# ============================================================
# CLEANUP
# ============================================================

finally:

    imu_fp.close()
    ppg_fp.close()
    vital_fp.close()
    env_fp.close()

    sock.close()

    print()
    print("=" * 65)
    print("                 FILES SAVED")
    print("=" * 65)

    print()
    print(imu_file)
    print(ppg_file)
    print(vital_file)
    print(env_file)

    print()
    print("=" * 65)
    print("                 FINAL COUNTS")
    print("=" * 65)

    print(
        f"IMU:         {imu_count}"
    )

    print(
        f"PPG:         {ppg_count}"
    )

    print(
        f"Vitals:      {vital_count}"
    )

    print(
        f"Environment: {env_count}"
    )

    print(
        f"Unknown:     {unknown_count}"
    )

    print()