from astropy.utils.data import download_file
from poliastro.data.iers import IERS_B_URL, IERS_B_FILENAME

print("[+] Forcing download of latest Earth Orientation Parameters (EOP)...")

try:
    # This function from astropy will download the file and cache it correctly
    # The poliastro library (used by gsopt) will then find and use it automatically
    download_file(IERS_B_URL, cache=True, show_progress=False)
    print("[+] SUCCESS: EOP data has been updated.")
    print("[+] You may now re-run the example script.")
except Exception as e:
    print(f"[!] FAILED: Could not download EOP data. Error: {e}")
    print("[!] Proceed to the fallback solution.")