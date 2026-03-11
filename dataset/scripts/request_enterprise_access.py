#!/usr/bin/env python3
"""
Generate a structured Enterprise dataset access request.
Use the output to send via email or paste into a contact form.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Request Enterprise 4DGS dataset access")
    parser.add_argument("--company", type=str, required=True, help="Company or organization name")
    parser.add_argument("--contact", type=str, required=True, help="Contact email")
    parser.add_argument("--use_case", type=str, default="", help="Short description of intended use (e.g. model training, product name)")
    parser.add_argument("--subjects", type=str, default="", help="Approximate number of subjects or data volume needed (optional)")
    parser.add_argument("--output", type=str, default="", help="Write request to JSON file (optional)")
    args = parser.parse_args()

    request = {
        "company": args.company,
        "contact_email": args.contact,
        "intended_use": args.use_case or "Commercial AI model training",
        "estimated_scale": args.subjects or "To be discussed",
    }

    text = (
        "Enterprise Dataset Access Request\n"
        "=================================\n\n"
        f"Company: {request['company']}\n"
        f"Contact: {request['contact_email']}\n"
        f"Intended use: {request['intended_use']}\n"
        f"Estimated scale: {request['estimated_scale']}\n\n"
        "Send this information to the maintainers (see dataset/ENTERPRISE_DATASET.md for contact options)."
    )

    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(request, f, indent=2)
        print(f"\nRequest saved to: {args.output}")


if __name__ == "__main__":
    main()
