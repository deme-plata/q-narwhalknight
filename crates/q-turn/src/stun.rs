// STUN (RFC 5389) and TURN (RFC 5766) message codec.
//
// Message type field layout (14 usable bits, top 2 always 0):
//   bit 13..9  = method[11..7]
//   bit 8      = class[1]
//   bit 7..5   = method[6..4]
//   bit 4      = class[0]
//   bit 3..0   = method[3..0]

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

pub const MAGIC_COOKIE: u32 = 0x2112_A442;
pub const HEADER_SIZE: usize = 20;

// ─── Message types ───────────────────────────────────────────────────────────
pub const MSG_BINDING_REQUEST:       u16 = 0x0001;
pub const MSG_BINDING_SUCCESS:       u16 = 0x0101;
pub const MSG_ALLOCATE_REQUEST:      u16 = 0x0003;
pub const MSG_ALLOCATE_SUCCESS:      u16 = 0x0103;
pub const MSG_ALLOCATE_ERROR:        u16 = 0x0113;
pub const MSG_REFRESH_REQUEST:       u16 = 0x0004;
pub const MSG_REFRESH_SUCCESS:       u16 = 0x0104;
pub const MSG_REFRESH_ERROR:         u16 = 0x0114;
pub const MSG_SEND_INDICATION:       u16 = 0x0016;
pub const MSG_DATA_INDICATION:       u16 = 0x0017;
pub const MSG_CREATE_PERM_REQUEST:   u16 = 0x0008;
pub const MSG_CREATE_PERM_SUCCESS:   u16 = 0x0108;
pub const MSG_CHANNEL_BIND_REQUEST:  u16 = 0x0009;
pub const MSG_CHANNEL_BIND_SUCCESS:  u16 = 0x0109;

// ─── Attribute types ──────────────────────────────────────────────────────────
pub const ATTR_MAPPED_ADDRESS:       u16 = 0x0001;
pub const ATTR_USERNAME:             u16 = 0x0006;
pub const ATTR_MESSAGE_INTEGRITY:    u16 = 0x0008;
pub const ATTR_ERROR_CODE:           u16 = 0x0009;
pub const ATTR_UNKNOWN_ATTRIBUTES:   u16 = 0x000A;
pub const ATTR_CHANNEL_NUMBER:       u16 = 0x000C;
pub const ATTR_LIFETIME:             u16 = 0x000D;
pub const ATTR_XOR_PEER_ADDRESS:     u16 = 0x0012;
pub const ATTR_DATA:                 u16 = 0x0016;
pub const ATTR_XOR_RELAYED_ADDRESS:  u16 = 0x0018;
pub const ATTR_REQUESTED_TRANSPORT:  u16 = 0x0019;
pub const ATTR_REALM:                u16 = 0x0014;
pub const ATTR_NONCE:                u16 = 0x0015;
pub const ATTR_XOR_MAPPED_ADDRESS:   u16 = 0x0020;
pub const ATTR_SOFTWARE:             u16 = 0x8022;
pub const ATTR_FINGERPRINT:          u16 = 0x8028;

// ─── Error codes ─────────────────────────────────────────────────────────────
pub const ERR_BAD_REQUEST:       (u16, &str) = (400, "Bad Request");
pub const ERR_UNAUTHORIZED:      (u16, &str) = (401, "Unauthorized");
pub const ERR_FORBIDDEN:         (u16, &str) = (403, "Forbidden");
pub const ERR_ALLOC_MISMATCH:    (u16, &str) = (437, "Allocation Mismatch");
pub const ERR_WRONG_CREDENTIALS: (u16, &str) = (441, "Wrong Credentials");
pub const ERR_UNSUPPORTED_PROTO: (u16, &str) = (442, "Unsupported Transport Protocol");
pub const ERR_QUOTA_REACHED:     (u16, &str) = (486, "Allocation Quota Reached");
pub const ERR_SERVER_ERROR:      (u16, &str) = (500, "Server Error");
pub const ERR_INSUF_CAPACITY:    (u16, &str) = (508, "Insufficient Capacity");

// UDP transport protocol number (REQUESTED-TRANSPORT value)
pub const TRANSPORT_UDP: u8 = 17;

/// A parsed STUN/TURN message.
///
/// `mi_offset` is the byte offset within the *original raw buffer* where the
/// MESSAGE-INTEGRITY attribute type field starts.  It is `Some` only when an
/// MI attribute was present and its 20-byte value was successfully extracted.
#[derive(Debug, Clone)]
pub struct StunMessage {
    pub msg_type:       u16,
    pub transaction_id: [u8; 12],
    /// Decoded attributes in order (type, raw-value-bytes without padding).
    /// MESSAGE-INTEGRITY is excluded — use `mi_value` instead.
    pub attrs: Vec<(u16, Vec<u8>)>,
    /// Byte offset in raw buffer where MI attr header starts (for HMAC-SHA1 verification)
    pub mi_offset: Option<usize>,
    /// The 20-byte HMAC value from the MESSAGE-INTEGRITY attribute
    pub mi_value: Option<[u8; 20]>,
}

impl StunMessage {
    pub fn new(msg_type: u16, txid: [u8; 12]) -> Self {
        Self {
            msg_type,
            transaction_id: txid,
            attrs: Vec::new(),
            mi_offset: None,
            mi_value: None,
        }
    }

    /// Parse a STUN message from the front of `buf`.
    /// Returns `(message, total_bytes_consumed)` on success.
    /// Returns `None` if the buffer doesn't start with a valid STUN packet.
    pub fn parse(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < HEADER_SIZE { return None; }
        let msg_type = u16::from_be_bytes([buf[0], buf[1]]);
        if msg_type & 0xC000 != 0 { return None; } // top 2 bits must be 0
        let attr_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
        let magic = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if magic != MAGIC_COOKIE { return None; }
        let total = HEADER_SIZE + attr_len;
        if buf.len() < total { return None; }

        let mut txid = [0u8; 12];
        txid.copy_from_slice(&buf[8..20]);

        let mut attrs    = Vec::new();
        let mut mi_offset = None;
        let mut mi_value  = None;
        let mut pos = HEADER_SIZE;

        while pos + 4 <= total {
            let atype = u16::from_be_bytes([buf[pos], buf[pos+1]]);
            let alen  = u16::from_be_bytes([buf[pos+2], buf[pos+3]]) as usize;
            pos += 4;
            if pos + alen > total { break; }
            let value = &buf[pos..pos+alen];

            if atype == ATTR_MESSAGE_INTEGRITY {
                if alen == 20 {
                    mi_offset = Some(pos - 4);
                    let mut v = [0u8; 20];
                    v.copy_from_slice(value);
                    mi_value = Some(v);
                }
                // Don't push MI into attrs — kept separate for integrity verification
            } else if atype == ATTR_FINGERPRINT {
                // Skip; we don't verify fingerprint on inbound but don't expose it
            } else {
                attrs.push((atype, value.to_vec()));
            }

            let padded = (alen + 3) & !3;
            pos += padded;
        }

        Some((Self { msg_type, transaction_id: txid, attrs, mi_offset, mi_value }, total))
    }

    /// Encode the message into bytes, adding MESSAGE-INTEGRITY and FINGERPRINT.
    /// `mi_key` is the HMAC-SHA1 key (16 bytes from MD5(user:realm:pass)).
    /// Pass `None` to skip MESSAGE-INTEGRITY (e.g. for 401 error responses).
    pub fn encode_with_integrity(
        &self,
        mi_key: Option<&[u8]>,
        software: &str,
    ) -> Vec<u8> {
        // Build attribute bytes (excluding MI and fingerprint)
        let mut attr_buf = self.build_attr_bytes();

        // Append SOFTWARE
        append_attr(&mut attr_buf, ATTR_SOFTWARE, software.as_bytes());

        if let Some(key) = mi_key {
            // Temporarily build the full message to compute MI over it
            // MI covers: header (with length set to current attrs + 24) + attr_buf
            let mi_len_field = (attr_buf.len() + 24) as u16; // 24 = 4-byte attr header + 20-byte HMAC
            let mut input = Vec::with_capacity(HEADER_SIZE + attr_buf.len());
            input.extend_from_slice(&self.msg_type.to_be_bytes());
            input.extend_from_slice(&mi_len_field.to_be_bytes());
            input.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
            input.extend_from_slice(&self.transaction_id);
            input.extend_from_slice(&attr_buf);

            let hmac = compute_hmac_sha1(key, &input);
            append_attr(&mut attr_buf, ATTR_MESSAGE_INTEGRITY, &hmac);
        }

        // Build final message (without fingerprint length yet)
        let mut out = Vec::with_capacity(HEADER_SIZE + attr_buf.len() + 8);
        // Reserve final length = attr_buf.len() + 8 (fingerprint attr)
        let final_len = (attr_buf.len() + 8) as u16;
        out.extend_from_slice(&self.msg_type.to_be_bytes());
        out.extend_from_slice(&final_len.to_be_bytes());
        out.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
        out.extend_from_slice(&self.transaction_id);
        out.extend_from_slice(&attr_buf);

        // Append FINGERPRINT (CRC32 of everything so far, XOR 0x5354554E)
        let crc = crc32fast::hash(&out) ^ 0x5354_554E;
        out.extend_from_slice(&ATTR_FINGERPRINT.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&crc.to_be_bytes());

        out
    }

    /// Encode without integrity (for 401 challenges where we have no key yet).
    pub fn encode_simple(&self, software: &str) -> Vec<u8> {
        self.encode_with_integrity(None, software)
    }

    fn build_attr_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (t, v) in &self.attrs {
            append_attr(&mut out, *t, v);
        }
        out
    }

    pub fn get_attr(&self, attr_type: u16) -> Option<&[u8]> {
        self.attrs.iter().find(|(t, _)| *t == attr_type).map(|(_, v)| v.as_slice())
    }

    pub fn add_xor_addr(&mut self, attr_type: u16, addr: SocketAddr) {
        self.attrs.push((attr_type, encode_xor_addr(addr, &self.transaction_id)));
    }

    pub fn add_u32(&mut self, attr_type: u16, val: u32) {
        self.attrs.push((attr_type, val.to_be_bytes().to_vec()));
    }

    pub fn add_bytes(&mut self, attr_type: u16, val: &[u8]) {
        self.attrs.push((attr_type, val.to_vec()));
    }

    pub fn add_str(&mut self, attr_type: u16, val: &str) {
        self.attrs.push((attr_type, val.as_bytes().to_vec()));
    }

    pub fn add_error(&mut self, code: u16, reason: &str) {
        let class  = (code / 100) as u8;
        let number = (code % 100) as u8;
        let mut v = vec![0u8, 0u8, class, number];
        v.extend_from_slice(reason.as_bytes());
        self.attrs.push((ATTR_ERROR_CODE, v));
    }
}

// ─── XOR address codec ────────────────────────────────────────────────────────

pub fn encode_xor_addr(addr: SocketAddr, txid: &[u8; 12]) -> Vec<u8> {
    let mut v = Vec::new();
    let xor_port = addr.port() ^ ((MAGIC_COOKIE >> 16) as u16);
    match addr.ip() {
        IpAddr::V4(ip) => {
            v.push(0x00);
            v.push(0x01); // IPv4
            v.extend_from_slice(&xor_port.to_be_bytes());
            let mc = MAGIC_COOKIE.to_be_bytes();
            for (b, x) in ip.octets().iter().zip(mc.iter()) {
                v.push(b ^ x);
            }
        }
        IpAddr::V6(ip) => {
            v.push(0x00);
            v.push(0x02); // IPv6
            v.extend_from_slice(&xor_port.to_be_bytes());
            let mut xk = [0u8; 16];
            xk[0..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            xk[4..16].copy_from_slice(txid);
            for (b, x) in ip.octets().iter().zip(xk.iter()) {
                v.push(b ^ x);
            }
        }
    }
    v
}

pub fn decode_xor_addr(data: &[u8], txid: &[u8; 12]) -> Option<SocketAddr> {
    if data.len() < 8 { return None; }
    let family = data[1];
    let port = u16::from_be_bytes([data[2], data[3]]) ^ ((MAGIC_COOKIE >> 16) as u16);
    match family {
        0x01 => {
            if data.len() < 8 { return None; }
            let mc = MAGIC_COOKIE.to_be_bytes();
            let ip = Ipv4Addr::new(
                data[4] ^ mc[0], data[5] ^ mc[1],
                data[6] ^ mc[2], data[7] ^ mc[3],
            );
            Some(SocketAddr::new(IpAddr::V4(ip), port))
        }
        0x02 => {
            if data.len() < 20 { return None; }
            let mut xk = [0u8; 16];
            xk[0..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            xk[4..16].copy_from_slice(txid);
            let mut ip = [0u8; 16];
            for i in 0..16 { ip[i] = data[4+i] ^ xk[i]; }
            Some(SocketAddr::new(IpAddr::V6(Ipv6Addr::from(ip)), port))
        }
        _ => None,
    }
}

// ─── HMAC-SHA1 (MESSAGE-INTEGRITY) ───────────────────────────────────────────

pub fn compute_hmac_sha1(key: &[u8], data: &[u8]) -> [u8; 20] {
    use ring::hmac;
    let k = hmac::Key::new(hmac::HMAC_SHA1_FOR_LEGACY_USE_ONLY, key);
    let tag = hmac::sign(&k, data);
    let mut out = [0u8; 20];
    out.copy_from_slice(tag.as_ref());
    out
}

/// Verify MESSAGE-INTEGRITY on an incoming message.
///
/// `raw` — the full original packet bytes.
/// `mi_offset` — byte offset of the MI attr header in `raw`.
/// `mi_value` — the 20 bytes from the MI attribute.
/// `key` — HMAC-SHA1 key = MD5(username ":" realm ":" password).
pub fn verify_message_integrity(
    raw: &[u8],
    mi_offset: usize,
    mi_value: &[u8; 20],
    key: &[u8],
) -> bool {
    if mi_offset < HEADER_SIZE || mi_offset + 24 > raw.len() {
        return false;
    }
    // Build HMAC input: all bytes before MI attr, with adjusted length field.
    // Length field = (offset of MI from end of header) + 24 (MI attr = 4-byte hdr + 20-byte value)
    let adj_len = (mi_offset - HEADER_SIZE + 24) as u16;
    let mut input = raw[0..mi_offset].to_vec();
    input[2] = (adj_len >> 8) as u8;
    input[3] = (adj_len & 0xFF) as u8;

    let expected = compute_hmac_sha1(key, &input);
    // Constant-time comparison to prevent timing attacks on MI verification
    expected.iter().zip(mi_value.iter()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

// ─── ChannelData ──────────────────────────────────────────────────────────────

/// Returns true if the packet starts with a ChannelData header (channel 0x4000–0x7FFF).
pub fn is_channel_data(buf: &[u8]) -> bool {
    if buf.len() < 4 { return false; }
    let ch = u16::from_be_bytes([buf[0], buf[1]]);
    ch >= 0x4000 && ch <= 0x7FFF
}

pub struct ChannelData<'a> {
    pub channel: u16,
    pub data:    &'a [u8],
}

pub fn parse_channel_data(buf: &[u8]) -> Option<ChannelData<'_>> {
    if buf.len() < 4 { return None; }
    let channel = u16::from_be_bytes([buf[0], buf[1]]);
    if channel < 0x4000 || channel > 0x7FFF { return None; }
    let length = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    if buf.len() < 4 + length { return None; }
    Some(ChannelData { channel, data: &buf[4..4+length] })
}

pub fn encode_channel_data(channel: u16, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + data.len());
    out.extend_from_slice(&channel.to_be_bytes());
    out.extend_from_slice(&(data.len() as u16).to_be_bytes());
    out.extend_from_slice(data);
    // Pad to 4-byte boundary on TCP (on UDP padding is not required)
    let pad = (4 - data.len() % 4) % 4;
    out.extend(std::iter::repeat(0u8).take(pad));
    out
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn append_attr(buf: &mut Vec<u8>, attr_type: u16, value: &[u8]) {
    buf.extend_from_slice(&attr_type.to_be_bytes());
    buf.extend_from_slice(&(value.len() as u16).to_be_bytes());
    buf.extend_from_slice(value);
    let pad = (4 - value.len() % 4) % 4;
    buf.extend(std::iter::repeat(0u8).take(pad));
}

/// Build a MAPPED-ADDRESS attribute value for a SocketAddr.
pub fn encode_mapped_address(addr: SocketAddr) -> Vec<u8> {
    let mut v = vec![0u8, 0u8]; // reserved
    match addr {
        SocketAddr::V4(a) => {
            v[1] = 0x01;
            v.extend_from_slice(&a.port().to_be_bytes());
            v.extend_from_slice(&a.ip().octets());
        }
        SocketAddr::V6(a) => {
            v[1] = 0x02;
            v.extend_from_slice(&a.port().to_be_bytes());
            v.extend_from_slice(&a.ip().octets());
        }
    }
    v
}
