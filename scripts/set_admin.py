#!/usr/bin/env python3
"""Grant or revoke the admin custom claim on a Firebase user.

Usage: python scripts/set_admin.py <email> [--revoke]
Needs Firebase Admin credentials (GOOGLE_APPLICATION_CREDENTIALS_JSON or ADC).
The user must sign out and in again for the claim to reach their token.
"""
from __future__ import annotations

import argparse

from firebase_admin import auth

from core.config import init_firebase_admin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--revoke", action="store_true")
    args = parser.parse_args()

    init_firebase_admin()
    user = auth.get_user_by_email(args.email)
    claims = dict(user.custom_claims or {})
    if args.revoke:
        claims.pop("admin", None)
    else:
        claims["admin"] = True
    auth.set_custom_user_claims(user.uid, claims)
    print(f"{'Revoked' if args.revoke else 'Granted'} admin for {args.email} ({user.uid})")


if __name__ == "__main__":
    main()
