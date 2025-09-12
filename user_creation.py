import argparse
from user_management import init_db, add_user


def main():
    parser = argparse.ArgumentParser(description="Create a new user for DocQuest")
    parser.add_argument("username", help="Username for the new account")
    parser.add_argument("password", help="Password for the new account")
    parser.add_argument(
        "--role", default="user", choices=["user", "admin"], help="Role of the user"
    )

    args = parser.parse_args()

    init_db()
    try:
        add_user(args.username, args.password, args.role)
        print(f"✅ User '{args.username}' created successfully with role '{args.role}'")
    except Exception as e:
        print(f"❌ Failed to create user: {e}")


if __name__ == "__main__":
    main()
