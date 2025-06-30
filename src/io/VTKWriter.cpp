// === src/io/VTKWriter.cpp ===
#include "cfd/io/VTKWriter.hpp"
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace cfd::io {

namespace fs = std::filesystem;

VTKWriter::VTKWriter(SharedPtr<Mesh> mesh, const std::string& outputDir)
    : mesh_(mesh), outputDir_(outputDir), writeCounter_(0) {
    
    // Create output directory if it doesn't exist
    fs::create_directories(outputDir_);
    
    // Determine format
    format_ = VTKFormat::ASCII; // Default to ASCII for compatibility
}

void VTKWriter::writeTimeStep(Real time,
                             const std::vector<FieldInfo>& fields) {
    std::stringstream filename;
    filename << outputDir_ << "/solution_" 
             << std::setfill('0') << std::setw(6) << writeCounter_
             << ".vtk";
    
    writeVTKFile(filename.str(), fields);
    
    // Update time series file
    updateTimeSeries(writeCounter_, time, filename.str());
    
    writeCounter_++;
}

void VTKWriter::writeVTKFile(const std::string& filename,
                            const std::vector<FieldInfo>& fields) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    writeHeader(file);
    
    // Write mesh
    writePoints(file);
    writeCells(file);
    
    // Write field data
    writeCellData(file, fields);
    
    file.close();
    
    logger_->debug("Wrote VTK file: {}", filename);
}

void VTKWriter::writeHeader(std::ofstream& file) {
    file << "# vtk DataFile Version 3.0\n";
    file << "CFD Solver Output\n";
    file << (format_ == VTKFormat::ASCII ? "ASCII" : "BINARY") << "\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
}

void VTKWriter::writePoints(std::ofstream& file) {
    const auto& vertices = mesh_->vertices();
    
    file << "POINTS " << vertices.size() << " double\n";
    
    for (const auto& vertex : vertices) {
        file << std::scientific << std::setprecision(6)
             << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
    }
}

void VTKWriter::writeCells(std::ofstream& file) {
    Index numCells = mesh_->numCells();
    Index totalSize = 0;
    
    // Count total size
    for (Index i = 0; i < numCells; ++i) {
        totalSize += 1 + mesh_->cell(i).vertices().size();
    }
    
    file << "\nCELLS " << numCells << " " << totalSize << "\n";
    
    // Write cell connectivity
    for (Index i = 0; i < numCells; ++i) {
        const Cell& cell = mesh_->cell(i);
        const auto& verts = cell.vertices();
        
        file << verts.size();
        for (Index v : verts) {
            file << " " << v;
        }
        file << "\n";
    }
    
    // Write cell types
    file << "\nCELL_TYPES " << numCells << "\n";
    
    for (Index i = 0; i < numCells; ++i) {
        const Cell& cell = mesh_->cell(i);
        file << getVTKCellType(cell.type()) << "\n";
    }
}

void VTKWriter::writeCellData(std::ofstream& file,
                             const std::vector<FieldInfo>& fields) {
    if (fields.empty()) return;
    
    file << "\nCELL_DATA " << mesh_->numCells() << "\n";
    
    for (const auto& [name, field] : fields) {
        if (auto scalarField = std::dynamic_pointer_cast<ScalarField>(field)) {
            writeScalarField(file, name, *scalarField);
        } else if (auto vectorField = std::dynamic_pointer_cast<VectorField>(field)) {
            writeVectorField(file, name, *vectorField);
        }
    }
}

void VTKWriter::writeScalarField(std::ofstream& file,
                                const std::string& name,
                                const ScalarField& field) {
    file << "\nSCALARS " << name << " double 1\n";
    file << "LOOKUP_TABLE default\n";
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        file << std::scientific << std::setprecision(6)
             << field[i] << "\n";
    }
}

void VTKWriter::writeVectorField(std::ofstream& file,
                                const std::string& name,
                                const VectorField& field) {
    file << "\nVECTORS " << name << " double\n";
    
    for (Index i = 0; i < mesh_->numCells(); ++i) {
        const Vector3& v = field[i];
        file << std::scientific << std::setprecision(6)
             << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
}

int VTKWriter::getVTKCellType(CellType type) const {
    switch (type) {
        case CellType::TETRAHEDRON:  return 10;
        case CellType::HEXAHEDRON:   return 12;
        case CellType::PRISM:        return 13;
        case CellType::PYRAMID:      return 14;
        case CellType::TRIANGLE:     return 5;
        case CellType::QUADRILATERAL: return 9;
        default:
            throw std::runtime_error("Unknown cell type");
    }
}

void VTKWriter::updateTimeSeries(int step, Real time, const std::string& filename) {
    std::string seriesFile = outputDir_ + "/solution.series";
    
    std::ofstream file(seriesFile, std::ios::app);
    if (!file) {
        logger_->warn("Cannot update time series file");
        return;
    }
    
    if (step == 0) {
        file << "# Time series data\n";
        file << "# Step Time Filename\n";
    }
    
    file << step << " " << time << " " << fs::path(filename).filename().string() << "\n";
}

// ParaView Data (PVD) writer for time series
void VTKWriter::writePVD() {
    std::string pvdFile = outputDir_ + "/solution.pvd";
    
    std::ofstream file(pvdFile);
    if (!file) {
        logger_->warn("Cannot create PVD file");
        return;
    }
    
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    file << "  <Collection>\n";
    
    // Read time series data
    std::string seriesFile = outputDir_ + "/solution.series";
    std::ifstream series(seriesFile);
    
    if (series) {
        std::string line;
        while (std::getline(series, line)) {
            if (line[0] == '#') continue;
            
            int step;
            Real time;
            std::string filename;
            std::istringstream iss(line);
            
            if (iss >> step >> time >> filename) {
                file << "    <DataSet timestep=\"" << time 
                     << "\" file=\"" << filename << "\"/>\n";
            }
        }
    }
    
    file << "  </Collection>\n";
    file << "</VTKFile>\n";
}
