// Q-NarwhalKnight RocksDB Encryption Provider
// v1.0.40-beta: C++ implementation of BlockAccessCipherStream
//
// ARCHITECTURE:
// - C++ wrapper around Rust AES-CTR implementation
// - FFI calls to Rust for actual encryption/decryption
// - Integrates with RocksDB EncryptionProvider API

#ifndef QNK_ENCRYPTION_PROVIDER_H
#define QNK_ENCRYPTION_PROVIDER_H

#include <rocksdb/env.h>
#include <rocksdb/env_encryption.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

namespace qnk {

// Forward declarations for Rust FFI
extern "C" {
    // Create AES-CTR stream cipher (calls Rust)
    void* qnk_create_cipher_stream(
        uint64_t file_id,
        uint32_t cf_id,
        const uint8_t* file_key,
        size_t key_len,
        const uint8_t* header,
        size_t header_len
    );

    // Destroy cipher stream
    void qnk_destroy_cipher_stream(void* stream);

    // Encrypt at offset (in-place)
    int qnk_encrypt_at_offset(
        void* stream,
        uint8_t* data,
        size_t len,
        uint64_t offset
    );

    // Decrypt at offset (in-place)
    int qnk_decrypt_at_offset(
        void* stream,
        uint8_t* data,
        size_t len,
        uint64_t offset
    );

    // Get file header from stream
    int qnk_get_file_header(
        void* stream,
        uint8_t* header_out,
        size_t header_size
    );

    // Derive file key from encryption manager (calls Rust)
    int qnk_derive_file_key(
        const char* keys_file_path,
        const char* passphrase,
        uint64_t file_id,
        uint32_t cf_id,
        uint8_t* key_out,
        size_t key_size
    );
}

/// AES-CTR cipher stream for RocksDB SST files
///
/// CRITICAL: This is a thin C++ wrapper around Rust implementation
/// All actual cryptography happens in Rust (memory-safe)
class AesCtrCipherStream : public rocksdb::BlockAccessCipherStream {
public:
    /// Create new cipher stream for encryption
    AesCtrCipherStream(uint64_t file_id, uint32_t cf_id, const std::string& file_key);

    /// Create cipher stream from existing header (decryption)
    AesCtrCipherStream(uint64_t file_id, uint32_t cf_id, const std::string& file_key,
                      const std::string& header);

    ~AesCtrCipherStream() override;

    // Disable copy/move
    AesCtrCipherStream(const AesCtrCipherStream&) = delete;
    AesCtrCipherStream& operator=(const AesCtrCipherStream&) = delete;

    /// Get block size (4KB for AES-CTR)
    size_t BlockSize() override {
        return 4096;
    }

    /// Encrypt data at given offset (in-place)
    rocksdb::Status Encrypt(uint64_t file_offset, char* data, size_t data_size) override;

    /// Decrypt data at given offset (in-place)
    rocksdb::Status Decrypt(uint64_t file_offset, char* data, size_t data_size) override;

    /// Get encryption header to write to file
    rocksdb::Status GetHeader(std::string* header) override;

private:
    void* rust_stream_;  // Opaque pointer to Rust AesCtrStream
    uint64_t file_id_;
    uint32_t cf_id_;
};

/// Q-NarwhalKnight encryption provider for RocksDB
///
/// RESPONSIBILITIES:
/// - Manage encryption keys (via Rust EncryptionManager)
/// - Create cipher streams for SST files and WAL
/// - Provide file headers for encrypted files
class QNarwhalEncryptionProvider : public rocksdb::EncryptionProvider {
public:
    /// Create encryption provider from keys file and passphrase
    QNarwhalEncryptionProvider(const std::string& keys_file_path,
                              const std::string& passphrase);

    ~QNarwhalEncryptionProvider() override = default;

    /// Get provider name
    const char* Name() const override {
        return "Q-NarwhalKnight-AES256-CTR";
    }

    /// Get header size (64 bytes)
    size_t GetPrefixLength() override {
        return 64;  // EncryptedFileHeader::SIZE
    }

    /// Create new prefix (header) for a new encrypted file
    rocksdb::Status CreateNewPrefix(const std::string& fname, char* prefix,
                                   size_t prefix_length) override;

    /// Create cipher stream for reading/writing encrypted file
    rocksdb::Status CreateCipherStream(
        const std::string& fname,
        const rocksdb::EnvOptions& options,
        rocksdb::Slice& prefix,
        std::unique_ptr<rocksdb::BlockAccessCipherStream>* result) override;

    /// Add cipher to provider (for key rotation)
    rocksdb::Status AddCipher(const std::string& descriptor, const char* cipher,
                             size_t len, bool for_write) override;

private:
    /// Extract file ID from filename
    uint64_t ExtractFileId(const std::string& fname);

    /// Extract column family ID from filename
    uint32_t ExtractCfId(const std::string& fname);

    /// Derive file encryption key
    rocksdb::Status DeriveFileKey(uint64_t file_id, uint32_t cf_id,
                                  std::string* key_out);

    std::string keys_file_path_;
    std::string passphrase_;
    std::mutex mutex_;  // Protect key derivation

    // File ID counter for new files
    std::atomic<uint64_t> next_file_id_{1};

    // Cache of file IDs (filename -> file_id)
    std::unordered_map<std::string, uint64_t> file_id_cache_;
};

} // namespace qnk

#endif // QNK_ENCRYPTION_PROVIDER_H
