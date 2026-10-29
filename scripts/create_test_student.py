"""
scripts/create_test_student.py
--------------------------------
Interactively creates a student (role='employee') login account for live
testing. No password is ever hardcoded -- you type it here, interactively.

This only creates the LOGIN. It does NOT fabricate a face enrollment --
face_registered stays 0, so after logging in at /login you still register
your own face for real via the webcam (Dashboard -> Register Face), the same
way any real student would. That's the only way to get a genuine face
embedding of your own face into the system.

Usage:
    python scripts/create_test_student.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werkzeug.security import generate_password_hash
from database.db import get_db_connection, init_db
from utils.validators import (
    validate_email, validate_password, validate_name,
    validate_employee_id, validate_phone, validate_department,
    ALLOWED_DEPARTMENTS,
)


def main():
    init_db()

    print("=== Create a Test Student Account ===")
    print(f"Allowed departments: {', '.join(ALLOWED_DEPARTMENTS)}\n")

    name, err = validate_name((input("Student name: ").strip()))
    if err:
        print(f"  {err}"); sys.exit(1)

    emp_id, err = validate_employee_id(input("Student ID (e.g. STU001): ").strip())
    if err:
        print(f"  {err}"); sys.exit(1)

    email, err = validate_email(input("Email (must be a real, deliverable address): ").strip())
    if err:
        print(f"  {err}"); sys.exit(1)

    phone, err = validate_phone(input("Phone (10-15 digits): ").strip())
    if err:
        print(f"  {err}"); sys.exit(1)

    department, err = validate_department(input(f"Department [{ALLOWED_DEPARTMENTS[0]}]: ").strip() or ALLOWED_DEPARTMENTS[0])
    if err:
        print(f"  {err}"); sys.exit(1)

    import getpass
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("  Passwords do not match."); sys.exit(1)
    password, err = validate_password(password)
    if err:
        print(f"  {err}"); sys.exit(1)

    conn = get_db_connection()
    existing = conn.execute(
        "SELECT id FROM users WHERE email = ? OR employee_id = ?", (email, emp_id)
    ).fetchone()
    if existing:
        conn.close()
        print(f"  A user with that email or student ID already exists.")
        sys.exit(1)

    conn.execute('''
        INSERT INTO users (employee_id, name, email, phone, department, password_hash, role, face_registered)
        VALUES (?, ?, ?, ?, ?, ?, 'employee', 0)
    ''', (emp_id, name, email, phone, department, generate_password_hash(password)))
    conn.commit()
    conn.close()

    print(f"\n  Student account created: {email} ({emp_id})")
    print("  Log in at /login with this email + password, then go to the")
    print("  Dashboard and register your face for real via the webcam --")
    print("  that step cannot be scripted, it needs your actual face.")


if __name__ == "__main__":
    main()
