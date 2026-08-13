"""Self-signed cert must satisfy Apple's TLS server-cert rules, or iOS refuses
the connection before fingerprint pinning ever runs."""

from __future__ import annotations

import datetime

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID

from pupa_backend import tls_check
from pupa_backend.scripts import setup as wizard


@pytest.fixture
def generated_cert(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard, "TLS_DIR", tmp_path / "tls")
    cert_path, _key_path, fingerprint = wizard._generate_tls_cert("host.example.ts.net")
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    return cert_path, cert, fingerprint


def test_validity_within_apple_limit(generated_cert):
    _path, cert, _fp = generated_cert
    lifetime = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
    assert lifetime <= 398


def test_has_server_auth_eku_and_hostname_san(generated_cert):
    _path, cert, _fp = generated_cert
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "host.example.ts.net" in san.get_values_for_type(x509.DNSName)


def test_fingerprint_matches_der_sha256(generated_cert):
    _path, cert, fingerprint = generated_cert
    import hashlib

    der = cert.public_bytes(serialization.Encoding.DER)
    assert fingerprint == hashlib.sha256(der).hexdigest()


def _write_cert(path, *, days: int, expired_days_ago: int = 0):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "t")])
    end = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=expired_days_ago)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(end - datetime.timedelta(days=days))
        .not_valid_after(end)
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return path


def test_startup_warns_about_over_long_cert(tmp_path, caplog):
    path = _write_cert(tmp_path / "long.crt", days=3650, expired_days_ago=-3600)
    tls_check.warn_unusable_cert(str(path))
    assert "398" in caplog.text


def test_startup_quiet_for_compliant_cert(tmp_path, caplog):
    path = _write_cert(tmp_path / "ok.crt", days=397, expired_days_ago=-300)
    tls_check.warn_unusable_cert(str(path))
    assert caplog.text == ""
