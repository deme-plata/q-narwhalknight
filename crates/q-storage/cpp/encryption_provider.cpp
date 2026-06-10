// Q-NarwhalKnight RocksDB Encryption Provider Implementation
// v1.0.40-beta: C++ wrapper around Rust AES-CTR encryption

#include "encryption_provider.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace qnk {

//==============================================================================
// AesCtrCipherStream Implementation
//==============================================================================

AesCtrCipherStream::AesCtrCipherStream(uint64_t file_id, uint32_t cf_id,
                                       const std::string& file_key)
    : rust_stream_(nullptr), file_id_(file_id), cf_id_(cf_id) {

    // Create new cipher stream (encryption mode)
    rust_stream_ = qnk_create_cipher_stream(
        file_id,
        cf_id,
        reinterpret_cast<const uint8_t*>(file_key.data()),
        file_key.size(),
        nullptr,  // No existing header
        0
    );

    if (!rust_stream_) {
        throw std::runtime_error("Failed to create AES-CTR cipher stream");
    }
}

AesCtrCipherStream::AesCtrCipherStream(uint64_t file_id, uint32_t cf_id,
                                       const std::string& file_key,
                                       const std::string& header)
    : rust_stream_(nullptr), file_id_(file_id), cf_id_(cf_id) {

    // Load existing cipher stream (decryption mode)
    rust_stream_ = qnk_create_cipher_stream(
        file_id,
        cf_id,
        reinterpret_cast<const uint8_t*>(file_key.data()),
        file_key.size(),
        reinterpret_cast<const uint8_t*>(header.data()),
        header.size()
    );

    if (!rust_stream_) {
        throw std::runtime_error("Failed to load AES-CTR cipher stream from header");
    }
}

AesCtrCipherStream::~AesCtrCipherStream() {
    if (rust_stream_) {
        qnk_destroy_cipher_stream(rust_stream_);
        rust_stream_ = nullptr;
    }
}

rocksdb::Status AesCtrCipherStream::Encrypt(uint64_t file_offset,
                                            char* data,
                                            size_t data_size) {
    if (!rust_stream_) {
        return rocksdb::Status::InvalidArgument("Cipher stream not initialized");
    }

    int result = qnk_encrypt_at_offset(
        rust_stream_,
        reinterpret_cast<uint8_t*>(data),
        data_size,
        file_offset
    );

    if (result != 0) {
        std::stringstream ss;
        ss << "Encryption failed at offset " << file_offset
           << " (error code: " << result << ")";
        return rocksdb::Status::IOError(ss.str());
    }

    return rocksdb::Status::OK();
}

rocksdb::Status AesCtrCipherStream::Decrypt(uint64_t file_offset,
                                            char* data,
                                            size_t data_size) {
    if (!rust_stream_) {
        return rocksdb::Status::InvalidArgument("Cipher stream not initialized");
    }

    int result = qnk_decrypt_at_offset(
        rust_stream_,
        reinterpret_cast<uint8_t*>(data),
        data_size,
        file_offset
    );

    if (result != 0) {
        std::stringstream ss;
        ss << "Decryption failed at offset " << file_offset
           << " (error code: " << result << ")";
        return rocksdb::Status::IOError(ss.str());
    }

    return rocksdb::Status::OK();
}

rocksdb::Status AesCtrCipherStream::GetHeader(std::string* header) {
    if (!rust_stream_) {
        return rocksdb::Status::InvalidArgument("Cipher stream not initialized");
    }

    // Allocate buffer for 64-byte header
    uint8_t header_buf[64];

    int result = qnk_get_file_header(
        rust_stream_,
        header_buf,
        sizeof(header_buf)
    );

    if (result != 0) {
        return rocksdb::Status::IOError("Failed to get file header");
    }

    header->assign(reinterpret_cast<char*>(header_buf), sizeof(header_buf));
    return rocksdb::Status::OK();
}

//==============================================================================
// QNarwhalEncryptionProvider Implementation
//==============================================================================

QNarwhalEncryptionProvider::QNarwhalEncryptionProvider(
    const std::string& keys_file_path,
    const std::string& passphrase)
    : keys_file_path_(keys_file_path),
      passphrase_(passphrase) {

    // Verify keys file exists
    if (keys_file_path.empty()) {
        throw std::invalid_argument("Keys file path cannot be empty");
    }
    if (passphrase.empty()) {
        throw std::invalid_argument("Passphrase cannot be empty");
    }
}

rocksdb::Status QNarwhalEncryptionProvider::CreateNewPrefix(
    const std::string& fname,
    char* prefix,
    size_t prefix_length) {

    if (prefix_length < 64) {
        return rocksdb::Status::InvalidArgument(
            "Prefix length must be at least 64 bytes");
    }

    // Extract file ID and CF ID from filename
    uint64_t file_id = ExtractFileId(fname);
    uint32_t cf_id = ExtractCfId(fname);

    // Derive file encryption key
    std::string file_key;
    rocksdb::Status s = DeriveFileKey(file_id, cf_id, &file_key);
    if (!s.ok()) {
        return s;
    }

    // Create temporary cipher stream to get header
    try {
        AesCtrCipherStream stream(file_id, cf_id, file_key);
        std::string header;
        s = stream.GetHeader(&header);
        if (!s.ok()) {
            return s;
        }

        // Copy header to prefix
        std::memcpy(prefix, header.data(), std::min(header.size(), prefix_length));
        return rocksdb::Status::OK();

    } catch (const std::exception& e) {
        return rocksdb::Status::IOError("Failed to create prefix: " + std::string(e.what()));
    }
}

rocksdb::Status QNarwhalEncryptionProvider::CreateCipherStream(
    const std::string& fname,
    const rocksdb::EnvOptions& options,
    rocksdb::Slice& prefix,
    std::unique_ptr<rocksdb::BlockAccessCipherStream>* result) {

    if (prefix.size() < 64) {
        return rocksdb::Status::InvalidArgument(
            "Prefix must be at least 64 bytes (EncryptedFileHeader)");
    }

    // Extract file ID and CF ID from header (or filename if not in header)
    uint64_t file_id = ExtractFileId(fname);
    uint32_t cf_id = ExtractCfId(fname);

    // Derive file encryption key
    std::string file_key;
    rocksdb::Status s = DeriveFileKey(file_id, cf_id, &file_key);
    if (!s.ok()) {
        return s;
    }

    // Create cipher stream from existing header
    try {
        std::string header(prefix.data(), prefix.size());
        auto stream = std::make_unique<AesCtrCipherStream>(
            file_id, cf_id, file_key, header);

        *result = std::move(stream);
        return rocksdb::Status::OK();

    } catch (const std::exception& e) {
        return rocksdb::Status::IOError(
            "Failed to create cipher stream: " + std::string(e.what()));
    }
}

rocksdb::Status QNarwhalEncryptionProvider::AddCipher(
    const std::string& descriptor,
    const char* cipher,
    size_t len,
    bool for_write) {

    // Currently not implementing key rotation
    // Will be added in Week 4
    return rocksdb::Status::NotSupported("Key rotation not yet implemented");
}

uint64_t QNarwhalEncryptionProvider::ExtractFileId(const std::string& fname) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache first
    auto it = file_id_cache_.find(fname);
    if (it != file_id_cache_.end()) {
        return it->second;
    }

    // Assign new file ID
    uint64_t file_id = next_file_id_.fetch_add(1);
    file_id_cache_[fname] = file_id;

    return file_id;
}

uint32_t QNarwhalEncryptionProvider::ExtractCfId(const std::string& fname) {
    // Try to extract CF ID from filename
    // Format: <number>.sst or <cf_name>/<number>.sst

    size_t slash_pos = fname.rfind('/');
    std::string basename = (slash_pos != std::string::npos)
        ? fname.substr(slash_pos + 1)
        : fname;

    // Check for column family prefix
    if (fname.find("/blocks/") != std::string::npos) {
        return 1;  // CF_BLOCKS
    } else if (fname.find("/dag_vertices/") != std::string::npos) {
        return 2;  // CF_DAG_VERTICES
    } else if (fname.find("/certificates/") != std::string::npos) {
        return 3;  // CF_CERTIFICATES
    }

    // Default CF
    return 0;
}

rocksdb::Status QNarwhalEncryptionProvider::DeriveFileKey(
    uint64_t file_id,
    uint32_t cf_id,
    std::string* key_out) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Allocate buffer for 32-byte key
    uint8_t key_buf[32];

    // Call Rust FFI to derive key
    int result = qnk_derive_file_key(
        keys_file_path_.c_str(),
        passphrase_.c_str(),
        file_id,
        cf_id,
        key_buf,
        sizeof(key_buf)
    );

    if (result != 0) {
        return rocksdb::Status::IOError("Failed to derive file encryption key");
    }

    key_out->assign(reinterpret_cast<char*>(key_buf), sizeof(key_buf));

    // Zeroize key buffer
    std::memset(key_buf, 0, sizeof(key_buf));

    return rocksdb::Status::OK();
}

} // namespace qnk
