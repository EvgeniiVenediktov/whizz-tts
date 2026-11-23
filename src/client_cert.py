"""The edge's TLS client identity for its upstream (mutual TLS): a key made on
the edge, which never leaves it, and a self-signed certificate the upstream
trusts (one entry in its trusted clients file; removing it revokes the edge).

  python -m src.client_cert     # print the certificate (public) for the upstream's owner

Paths come from SPEAK_UPSTREAM_CERT / SPEAK_UPSTREAM_KEY; the server makes the
pair on its first start (src/server.py), this prints it (making it if missing).
"""
import datetime
import os
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

VALID_DAYS = 3650


def ensure(certfile: str, keyfile: str, name: str = "whizz-tts-edge") -> bool:
    """Make the key (0600) and its certificate unless both exist -> True when made."""
    if os.path.exists(certfile) and os.path.exists(keyfile):
        return False
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    for path in (certfile, keyfile):
        os.makedirs(os.path.dirname(path) or ".", mode=0o700, exist_ok=True)
    fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return True


if __name__ == "__main__":
    cert, key = os.getenv("SPEAK_UPSTREAM_CERT"), os.getenv("SPEAK_UPSTREAM_KEY")
    if not (cert and key):
        sys.exit("set SPEAK_UPSTREAM_CERT and SPEAK_UPSTREAM_KEY")
    ensure(cert, key)
    with open(cert, encoding="ascii") as f:
        sys.stdout.write(f.read())
