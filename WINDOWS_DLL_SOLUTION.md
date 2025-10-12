# Windows DLL Solution for Q-NarwhalKnight

## Problem
The Windows executable (`q-api-server.exe`) requires mingw-w64 runtime DLLs:
- `libgfortran-5.dll` - Fortran runtime (required by OpenBLAS)
- `libgcc_s_seh-1.dll` - GCC exception handling
- `libquadmath-0.dll` - Quadruple precision math
- `libwinpthread-1.dll` - POSIX threading (✅ extracted)

## Solution Options

### Option 1: Bundle DLLs (Recommended)
Download the required DLLs and package them with the executable.

**Sources for DLLs**:
1. **MSYS2 mingw-w64 packages** (most reliable):
   ```
   https://repo.msys2.org/mingw/mingw64/mingw-w64-x86_64-gcc-libs-*.pkg.tar.zst
   ```

2. **Direct download from GitHub mirrors**:
   - libgfortran-5.dll: ~2MB
   - libgcc_s_seh-1.dll: ~100KB
   - libquadmath-0.dll: ~300KB
   - libwinpthread-1.dll: ~600KB (✅ already extracted)

3. **Extract from local mingw-w64 installation** (if available on Windows build machine)

### Option 2: Instruct Users to Install Runtime
Create installation instructions for users to install mingw-w64 runtime:
```powershell
# Using winget (Windows Package Manager)
winget install -e --id MSYS2.MSYS2

# Or download from: https://www.msys2.org/
```

### Option 3: Statically Link Everything
Modify build configuration to use `musl` target or pure static linking:
- Switch to `x86_64-pc-windows-msvc` (requires Visual Studio)
- Use alternative linear algebra library without Fortran dependencies
- Replace OpenBLAS with pure Rust implementation

## Recommended Approach

**Bundle the DLLs** with clear licensing information:

### Step 1: Download Required DLLs
```bash
# From MSYS2 repository or Windows system with mingw-w64 installed
# Copy these 4 DLLs to the Windows package directory
```

### Step 2: Update Windows Package Structure
```
q-narwhalknight-windows/
├── q-api-server.exe          (81MB - main executable)
├── libgfortran-5.dll         (2MB)
├── libgcc_s_seh-1.dll       (100KB)
├── libquadmath-0.dll        (300KB)
├── libwinpthread-1.dll      (600KB) ✅
├── README-WINDOWS.txt        (setup instructions)
└── LICENSE-MINGW.txt         (mingw-w64 license - GPL/LGPL)
```

### Step 3: Add DLL Information to README
```
INCLUDED RUNTIME LIBRARIES:
- libgfortran-5.dll - GNU Fortran Runtime (GPLv3 + GCC Runtime Exception)
- libgcc_s_seh-1.dll - GCC Runtime (GPLv3 + GCC Runtime Exception)
- libquadmath-0.dll - GCC Quad-Precision Math (GPLv3 + GCC Runtime Exception)
- libwinpthread-1.dll - MinGW-w64 POSIX Threads (MIT/BSD)

These libraries are from the mingw-w64 project and are redistributed under their respective licenses.
```

## Current Status
- ✅ `libwinpthread-1.dll` extracted (607KB)
- ⏳ Need to obtain: libgfortran-5.dll, libgcc_s_seh-1.dll, libquadmath-0.dll

## Next Steps
1. Download remaining DLLs from MSYS2 repository
2. Add DLLs to Windows package directory
3. Update README with DLL information
4. Create new Windows zip package
5. Test on actual Windows machine

## Alternative: Pure Static Build
If DLL bundling is not acceptable, consider:
- Switching to alternative crypto/math libraries without Fortran dependencies
- Using `x86_64-pc-windows-gnu` with fully static musl
- Building with MSVC toolchain instead of mingw-w64
