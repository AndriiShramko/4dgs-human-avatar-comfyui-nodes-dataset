# Enterprise Dataset

The **Enterprise** tier provides access to large-scale human scan libraries for training and fine-tuning your own AI models in a **commercial** setting.

## What you get

- **Volume:** Thousands of high-quality human scans, hundreds of terabytes of data.
- **Use:** Training and fine-tuning of AI models (including 4DGS and related neural rendering models).
- **License:** Commercial use permitted under a separate license agreement.
- **Delivery:** Access via S3-compatible storage or CDN; credentials and endpoints provided after agreement.

## Who it is for

- Companies building commercial avatar or 3D/4D human reconstruction products.
- Teams that need large, diverse human scan data to train proprietary models.
- Enterprise customers who require a formal license and support.

## How to request access

1. **Option A — Script (form-style):**  
   Run the request helper to generate a structured request (e.g. for email or web form):
   ```bash
   python dataset/scripts/request_enterprise_access.py --company "Your Company" --contact "you@company.com"
   ```

2. **Option B — Manual:**  
   Open an issue or contact the maintainers (contact details to be added) with:
   - Company / organization name
   - Contact email
   - Intended use (e.g. model training, product name)
   - Approximate data volume or number of subjects needed (if known)

3. **Next steps:**  
   You will receive a commercial license offer and, after signing, access credentials and download instructions (e.g. S3/CDN).

## License

Use of the Enterprise dataset is subject to a **separate commercial license**. See [DATASET_LICENSE_ENTERPRISE.md](../DATASET_LICENSE_ENTERPRISE.md) for an overview. No rights are granted by this repository alone.
