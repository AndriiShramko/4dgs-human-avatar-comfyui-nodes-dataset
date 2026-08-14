#!/usr/bin/env bash
set -euo pipefail

urle () {
  [[ "${1}" ]] || return 1
  local LANG=C i x
  for (( i = 0; i < ${#1}; i++ )); do
    x="${1:i:1}"
    [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
  done
  echo
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/src/models"
mkdir -p "${MODELS_DIR}"

command -v wget >/dev/null || { echo "wget is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }
command -v gdown >/dev/null || { echo "gdown is required. Install with: pip install gdown" >&2; exit 1; }

echo "Downloading MultiHMR checkpoint..."
mkdir -p "${MODELS_DIR}/multiHMR"
wget https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt \
  -O "${MODELS_DIR}/multiHMR/multiHMR_896_L.pt" \
  --no-check-certificate \
  --continue

echo
read -p "Username (SMPL-X, https://smpl-x.is.tue.mpg.de): " smplx_username
read -s -p "Password (SMPL-X): " smplx_password
echo
smplx_username="$(urle "${smplx_username}")"
smplx_password="$(urle "${smplx_password}")"

echo "Downloading SMPL-X models..."
mkdir -p "${MODELS_DIR}/smplx"
wget --post-data "username=${smplx_username}&password=${smplx_password}" \
  'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' \
  -O "${MODELS_DIR}/smplx.zip" \
  --no-check-certificate \
  --continue
unzip -q "${MODELS_DIR}/smplx.zip" -d "${MODELS_DIR}/smplx_tmp"
cp "${MODELS_DIR}"/smplx_tmp/models/smplx/* "${MODELS_DIR}/smplx/"
rm -rf "${MODELS_DIR}/smplx_tmp" "${MODELS_DIR}/smplx.zip"

echo
read -p "Username (SMPL, https://smpl.is.tue.mpg.de): " smpl_username
read -s -p "Password (SMPL): " smpl_password
echo
smpl_username="$(urle "${smpl_username}")"
smpl_password="$(urle "${smpl_password}")"

echo "Downloading SMPL models..."
mkdir -p "${MODELS_DIR}/smpl"
wget --post-data "username=${smpl_username}&password=${smpl_password}" \
  'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' \
  -O "${MODELS_DIR}/smpl/smpl.zip" \
  --no-check-certificate \
  --continue
unzip -q "${MODELS_DIR}/smpl/smpl.zip" -d "${MODELS_DIR}/smpl/smpl_tmp"
cp "${MODELS_DIR}/smpl/smpl_tmp/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" "${MODELS_DIR}/smpl/SMPL_NEUTRAL.pkl"
cp "${MODELS_DIR}/smpl/smpl_tmp/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl" "${MODELS_DIR}/smpl/SMPL_FEMALE.pkl"
cp "${MODELS_DIR}/smpl/smpl_tmp/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl" "${MODELS_DIR}/smpl/SMPL_MALE.pkl"
rm -rf "${MODELS_DIR}/smpl/smpl_tmp" "${MODELS_DIR}/smpl/smpl.zip"

echo "Downloading supplementary body-model files..."
TMP_SUPP="${MODELS_DIR}/_supplementary"
rm -rf "${TMP_SUPP}"
mkdir -p "${TMP_SUPP}"
gdown --folder -O "${TMP_SUPP}" 'https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7?usp=share_link'

mkdir -p "${MODELS_DIR}/body_models"
find "${TMP_SUPP}" -type f -name 'J_regressor_h36m.npy' -exec cp {} "${MODELS_DIR}/body_models/J_regressor_h36m.npy" \;
find "${TMP_SUPP}" -type f -name 'J_regressor_h36m.npy' -exec cp {} "${MODELS_DIR}/smpl/J_regressor_h36m.npy" \;
find "${TMP_SUPP}" -type f -name 'smpl_mean_params.npz' -exec cp {} "${MODELS_DIR}/body_models/smpl_mean_params.npz" \;
find "${TMP_SUPP}" -type f -name 'smplx2smpl.pkl' -exec cp {} "${MODELS_DIR}/body_models/smplx2smpl.pkl" \;
find "${TMP_SUPP}" -type f -name 'smplx2smpl.pkl' -exec cp {} "${MODELS_DIR}/smplx/smplx2smpl.pkl" \;
find "${TMP_SUPP}" -type f -name 'smplx2smpl_joints.npy' -exec cp {} "${MODELS_DIR}/body_models/smplx2smpl_joints.npy" \;
rm -rf "${TMP_SUPP}"

echo "Done. Body model assets are in ${MODELS_DIR}."
