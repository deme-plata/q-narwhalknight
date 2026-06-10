//! Visualization output in VTK format for ParaView/VisIt

use crate::field::ScalarField3D;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use vtkio::model::*;
use vtkio::*;
use std::io::BufWriter;

/// Export field to VTK format for visualization
pub struct VtkExporter;

impl VtkExporter {
    /// Export scalar field to VTK structured points format
    pub fn export_scalar_field<P: AsRef<Path>>(
        field: &ScalarField3D,
        path: P,
        field_name: &str,
    ) -> Result<()> {
        let res = field.resolution;

        // Create VTK ImageData (structured points)
        let extent = Extent::Ranges([0..=res as i32 - 1, 0..=res as i32 - 1, 0..=res as i32 - 1]);

        let spacing = [field.dx_nm, field.dx_nm, field.dx_nm];

        // Convert field data to flat vector
        let scalar_data: Vec<f64> = field
            .data
            .iter()
            .copied()
            .collect();

        // Create scalar attribute using DataArray
        let scalars = Attribute::DataArray(DataArray {
            name: field_name.to_string(),
            elem: ElementType::Scalars { num_comp: 1, lookup_table: None },
            data: IOBuffer::F64(scalar_data),
        });

        // Create point data
        let point_data = Attributes {
            point: vec![scalars],
            cell: vec![],
        };

        // Create VTK dataset
        let vtk = Vtk {
            version: Version::new((4, 2)),
            title: format!("Higgs Field Simulation - {}", field_name),
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::ImageData {
                extent: extent.clone(),
                origin: [0.0, 0.0, 0.0],
                spacing: [spacing[0] as f32, spacing[1] as f32, spacing[2] as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(ImageDataPiece {
                    extent,
                    data: point_data,
                }))],
            },
            file_path: None,
        };

        // Write to file
        vtk.export_ascii(path.as_ref())
            .context("Failed to write VTK file")?;

        Ok(())
    }

    /// Export vector field (e.g., gradient) to VTK
    pub fn export_vector_field<P: AsRef<Path>>(
        field_x: &ScalarField3D,
        field_y: &ScalarField3D,
        field_z: &ScalarField3D,
        path: P,
        field_name: &str,
    ) -> Result<()> {
        let res = field_x.resolution;

        let extent = Extent::Ranges([0..=res as i32 - 1, 0..=res as i32 - 1, 0..=res as i32 - 1]);

        // Interleave vector components
        let mut vector_data = Vec::with_capacity(res * res * res * 3);
        for ((x, y), z) in field_x
            .data
            .iter()
            .zip(field_y.data.iter())
            .zip(field_z.data.iter())
        {
            vector_data.push(*x);
            vector_data.push(*y);
            vector_data.push(*z);
        }

        // Create vector attribute
        let vectors = Attribute::DataArray(DataArray {
            name: field_name.to_string(),
            elem: ElementType::Vectors,
            data: IOBuffer::F64(vector_data),
        });

        let point_data = Attributes {
            point: vec![vectors],
            cell: vec![],
        };

        let spacing = [field_x.dx_nm, field_x.dx_nm, field_x.dx_nm];

        let vtk = Vtk {
            version: Version::new((4, 2)),
            title: format!("Higgs Field Gradient - {}", field_name),
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::ImageData {
                extent: extent.clone(),
                origin: [0.0, 0.0, 0.0],
                spacing: [spacing[0] as f32, spacing[1] as f32, spacing[2] as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(ImageDataPiece {
                    extent,
                    data: point_data,
                }))],
            },
            file_path: None,
        };

        vtk.export_ascii(path.as_ref())
            .context("Failed to write VTK file")?;

        Ok(())
    }

    /// Export time series of fields
    pub fn export_time_series<P: AsRef<Path>>(
        fields: &[ScalarField3D],
        output_dir: P,
        base_name: &str,
    ) -> Result<()> {
        let dir = output_dir.as_ref();
        std::fs::create_dir_all(dir).context("Failed to create output directory")?;

        for (i, field) in fields.iter().enumerate() {
            let filename = dir.join(format!("{}_{:06}.vtk", base_name, i));
            Self::export_scalar_field(field, filename, base_name)?;
        }

        // Create PVD file for ParaView time series
        Self::create_pvd_file(dir, base_name, fields.len())?;

        Ok(())
    }

    /// Create ParaView collection file (.pvd) for time series
    fn create_pvd_file<P: AsRef<Path>>(
        output_dir: P,
        base_name: &str,
        num_files: usize,
    ) -> Result<()> {
        let pvd_path = output_dir.as_ref().join(format!("{}.pvd", base_name));
        let mut file = File::create(&pvd_path).context("Failed to create PVD file")?;

        writeln!(file, "<?xml version=\"1.0\"?>")?;
        writeln!(
            file,
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">"
        )?;
        writeln!(file, "  <Collection>")?;

        for i in 0..num_files {
            writeln!(
                file,
                "    <DataSet timestep=\"{}\" group=\"\" part=\"0\" file=\"{}_{:06}.vtk\"/>",
                i, base_name, i
            )?;
        }

        writeln!(file, "  </Collection>")?;
        writeln!(file, "</VTKFile>")?;

        Ok(())
    }

    /// Export 2D slice of field
    pub fn export_2d_slice<P: AsRef<Path>>(
        field: &ScalarField3D,
        z_index: usize,
        path: P,
        field_name: &str,
    ) -> Result<()> {
        let slice = field
            .slice_xy(z_index)
            .context("Invalid slice index")?;

        let res = slice.nrows();

        // Create 2D VTK dataset
        let extent = Extent::Ranges([0..=res as i32 - 1, 0..=res as i32 - 1, 0..=0]);

        let scalar_data: Vec<f64> = slice.iter().copied().collect();

        let scalars = Attribute::DataArray(DataArray {
            name: field_name.to_string(),
            elem: ElementType::Scalars { num_comp: 1, lookup_table: None },
            data: IOBuffer::F64(scalar_data),
        });

        let point_data = Attributes {
            point: vec![scalars],
            cell: vec![],
        };

        let spacing = [field.dx_nm, field.dx_nm, field.dx_nm];

        let vtk = Vtk {
            version: Version::new((4, 2)),
            title: format!("Higgs Field 2D Slice - {}", field_name),
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::ImageData {
                extent: extent.clone(),
                origin: [0.0, 0.0, 0.0],
                spacing: [spacing[0] as f32, spacing[1] as f32, spacing[2] as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(ImageDataPiece {
                    extent,
                    data: point_data,
                }))],
            },
            file_path: None,
        };

        vtk.export_ascii(path.as_ref())
            .context("Failed to write VTK file")?;

        Ok(())
    }
}

/// Simple ASCII export for quick visualization
pub struct AsciiExporter;

impl AsciiExporter {
    /// Export central slice to simple text format
    pub fn export_slice_txt<P: AsRef<Path>>(
        field: &ScalarField3D,
        z_index: usize,
        path: P,
    ) -> Result<()> {
        let slice = field
            .slice_xy(z_index)
            .context("Invalid slice index")?;

        let mut file = File::create(path.as_ref()).context("Failed to create file")?;

        writeln!(file, "# Higgs field slice at z_index = {}", z_index)?;
        writeln!(file, "# Resolution: {}x{}", slice.nrows(), slice.ncols())?;
        writeln!(file, "# Grid spacing: {} nm", field.dx_nm)?;
        writeln!(file, "#")?;

        for row in slice.rows() {
            for val in row {
                write!(file, "{:12.6e} ", val)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    /// Export 1D line profile through center
    pub fn export_line_profile<P: AsRef<Path>>(
        field: &ScalarField3D,
        path: P,
    ) -> Result<()> {
        let mut file = File::create(path.as_ref()).context("Failed to create file")?;

        writeln!(file, "# Higgs field line profile through center")?;
        writeln!(file, "# x(nm) phi")?;

        let center_y = field.resolution / 2;
        let center_z = field.resolution / 2;

        for i in 0..field.resolution {
            let x = i as f64 * field.dx_nm;
            let phi = field.get(i, center_y, center_z).unwrap_or(0.0);
            writeln!(file, "{:.6e} {:.6e}", x, phi)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vtk_scalar_export() {
        let field = ScalarField3D::new(16, 50.0);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_field.vtk");

        let result = VtkExporter::export_scalar_field(&field, &output_path, "phi");
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_2d_slice_export() {
        let mut field = ScalarField3D::new(16, 50.0);
        field.fill_uniform(246.22);

        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_slice.vtk");

        let result = VtkExporter::export_2d_slice(&field, 8, &output_path, "phi");
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_ascii_export() {
        let mut field = ScalarField3D::new(16, 50.0);
        field.fill_uniform(246.22);

        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_slice.txt");

        let result = AsciiExporter::export_slice_txt(&field, 8, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_line_profile() {
        let mut field = ScalarField3D::new(16, 50.0);
        field.fill_uniform(246.22);

        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("profile.txt");

        let result = AsciiExporter::export_line_profile(&field, &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }
}
