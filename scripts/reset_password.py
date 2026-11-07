"""
scripts/reset_password.py
---------------------------
Resets the password for an existing account (admin or student) by email.

Local/CLI only -- there is no web route for this. Passwords are hashed
(werkzeug's generate_password_hash); an existing hash can never be
recovered, only overwritten with a new one you set here.

Usage:
    python scripts/reset_password.py you@example.com
    python scripts/reset_password.py              # prompts for the email instead
"""

import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werkzeug.security import generate_password_hash
from database.db import get_db_connection
from utils.validators import validate_password


def main():
    email_input = sys.argv[1] if len(sys.argv) > 1 else input("Account email: ").strip()
    # Just normalize (matches how registration stores it) -- no live DNS check here.
    # This account's email was already validated when it was created; re-running that
    # check on every reset would make a transient DNS hiccup block you from resetting
    # your own password.
    email = email_input.strip().lower()
    if not email:
        print("  Email is required.")
        sys.exit(1)

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, name, role FROM users WHERE email = ?", (email,)
    ).fetchone()

    if not user:
        print(f"  No account found with email: {email}")
        conn.close()
        sys.exit(1)

    print(f"Found: {user['name']} ({user['role']}) -- {email}")
    confirm = input("Reset this account's password? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        conn.close()
        sys.exit(0)

    password = getpass.getpass("New password: ")
    confirm_pw = getpass.getpass("Confirm new password: ")
    if password != confirm_pw:
        print("  Passwords do not match.")
        conn.close()
        sys.exit(1)

    password, err = validate_password(password)
    if err:
        print(f"  {err}")
        conn.close()
        sys.exit(1)

    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(password), user["id"]),
    )
    conn.commit()
    conn.close()
    print(f"  Password reset for {email} ({user['role']}).")


if __name__ == "__main__":
    main()
