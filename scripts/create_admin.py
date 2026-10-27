"""
scripts/create_admin.py
------------------------
Interactively creates (or resets the password of) an admin account.

No admin credentials are ever hardcoded in source. If you'd rather automate
first-run setup (e.g. in a deploy script), set ADMIN_EMAIL, ADMIN_PASSWORD,
and optionally ADMIN_NAME in the environment before the app's first run
instead -- init_db() will create the admin from those automatically.

Usage:
    python scripts/create_admin.py
"""

import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werkzeug.security import generate_password_hash
from database.db import get_db_connection, init_db
from utils.validators import validate_email, validate_password, validate_name


def main():
    init_db()  # ensure tables exist; does NOT seed an admin without env vars

    print("=== SmartFace Admin Setup ===")
    name_input = input("Admin name [Admin]: ").strip() or "Admin"
    name, err = validate_name(name_input)
    if err:
        print(f"  {err}")
        sys.exit(1)

    email_input = input("Admin email: ").strip()
    email, err = validate_email(email_input)
    if err:
        print(f"  {err}")
        sys.exit(1)

    password = getpass.getpass("Admin password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("  Passwords do not match.")
        sys.exit(1)

    password, err = validate_password(password)
    if err:
        print(f"  {err}")
        sys.exit(1)

    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()

    if existing:
        conn.execute(
            "UPDATE users SET password_hash = ?, role = 'admin' WHERE id = ?",
            (generate_password_hash(password), existing["id"]),
        )
        conn.commit()
        conn.close()
        print(f"  Updated existing user {email} -> admin, password reset.")
    else:
        conn.execute('''
            INSERT INTO users (employee_id, name, email, phone, department, password_hash, role, face_registered)
            VALUES (?, ?, ?, ?, ?, ?, 'admin', 0)
        ''', (f"ADMIN{os.urandom(2).hex().upper()}", name, email, "0000000000", "Management",
              generate_password_hash(password)))
        conn.commit()
        conn.close()
        print(f"  Admin account created: {email}")


if __name__ == "__main__":
    main()
