#pragma once

#include "cfd/core/Types.hpp"
#include "cfd/core/Mesh.hpp"
#include "cfd/core/Field.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <map>

namespace cfd::io {

// VTK file writer for visualization
class VTKWriter {
public:
    enum class Format {
        ASCII,
        BINARY,
        XML  // Modern VTK XML format
    };
    
    enum class DataLocation {
        POINT_DATA,
        CELL_DATA
    };
    
    VTKWriter(const Mesh& mesh, const std::string& baseName = "output")
        : mesh_(mesh), baseName_(baseName), format_(Format::BINARY) {}
    
    // Set output format
    void setFormat(Format format) { format_ = format; }
    
    // Set compression for XML format
    void setCompression(bool compress) { compress_ = compress; }
    
    // Add fields to write
    void addField(const ScalarField& field, DataLocation location = DataLocation::CELL_DATA);
    void addField(const VectorField& field, DataLocation location = DataLocation::CELL_DATA);
    void addField(const TensorField& field, DataLocation location = DataLocation::CELL_DATA);
    
    // Write single time step
    void write(Real time);
    void write(const std::string& filename);
    
    // Write time series (creates .pvd file for ParaView)
    void writeTimeSeries(const std::vector<Real>& times);
    
    // Write mesh only
    void writeMesh(const std::string& filename);
    
    // Clear field list
    void clearFields() { fields_.clear(); }
    
private:
    const Mesh& mesh_;
    std::string baseName_;
    Format format_;
    bool compress_ = false;
    
    // Field data
    struct FieldData {
        std::string name;
        FieldType type;
        DataLocation location;
        const void* data;
    };
    std::vector<FieldData> fields_;
    
    // Time series data
    std::vector<std::pair<Real, std::string>> timeSeriesFiles_;
    
    // Legacy VTK format writers
    void writeLegacyASCII(const std::string& filename);
    void writeLegacyBinary(const std::string& filename);
    
    // XML VTK format writers
    void writeXMLUnstructuredGrid(const std::string& filename);
    void writeXMLPolyData(const std::string& filename);
    
    // Write mesh data
    void writeMeshLegacy(std::ofstream& file);
    void writeMeshXML(std::ofstream& file);
    
    // Write field data
    void writeFieldsLegacy(std::ofstream& file);
    void writeFieldsXML(std::ofstream& file);
    
    // Write PVD file for time series
    void writePVDFile();
    
    // Utility functions
    void writeHeader(std::ofstream& file);
    void writeCellType(std::ofstream& file, CellType type);
    int getVTKCellType(CellType type) const;
    
    // Binary write helpers
    template<typename T>
    void writeBinary(std::ofstream& file, const T& value);
    
    template<typename T>
    void writeBinaryArray(std::ofstream& file, const T* data, size_t count);
    
    // Base64 encoding for XML format
    std::string base64Encode(const void* data, size_t size);
};

// Parallel VTK writer using MPI
class ParallelVTKWriter : public VTKWriter {
public:
    ParallelVTKWriter(const Mesh& mesh, int rank, int size,
                     const std::string& baseName = "output")
        : VTKWriter(mesh, baseName), rank_(rank), size_(size) {}
    
    // Write parallel data
    void writeParallel(Real time);
    
    // Write parallel time series
    void writeParallelTimeSeries(const std::vector<Real>& times);
    
private:
    int rank_;
    int size_;
    
    // Write piece file
    void writePiece(const std::string& filename);
    
    // Write parallel summary file
    void writePVTU(const std::string& filename);
    
    // Gather metadata from all processes
    void gatherMetadata(std::vector<int>& pieceSizes);
};

// CGNS writer (optional, for more advanced users)
class CGNSWriter {
public:
    CGNSWriter(const Mesh& mesh, const std::string& filename)
        : mesh_(mesh), filename_(filename) {}
    
    // Write CGNS file
    void write();
    
    // Add solution
    void addSolution(const std::string& name, Real time);
    
    // Add field
    void addField(const ScalarField& field);
    void addField(const VectorField& field);
    
private:
    const Mesh& mesh_;
    std::string filename_;
    int fileId_ = -1;
    int baseId_ = -1;
    int zoneId_ = -1;
    
    // CGNS-specific functions
    void writeBase();
    void writeZone();
    void writeCoordinates();
    void writeElements();
    void writeBoundaryConditions();
};

// Ensight Gold format writer
class EnsightWriter {
public:
    EnsightWriter(const Mesh& mesh, const std::string& caseName)
        : mesh_(mesh), caseName_(caseName) {}
    
    // Initialize case
    void initialize();
    
    // Write time step
    void writeTimeStep(Real time);
    
    // Add fields
    void addField(const ScalarField& field);
    void addField(const VectorField& field);
    
    // Finalize case
    void finalize();
    
private:
    const Mesh& mesh_;
    std::string caseName_;
    int timeStep_ = 0;
    std::vector<Real> times_;
    std::vector<std::string> fieldNames_;
    
    // Write case file
    void writeCaseFile();
    
    // Write geometry
    void writeGeometry();
    
    // Write variable
    void writeVariable(const std::string& name, const void* data,
                      FieldType type, int timeStep);
};

// Tecplot writer
class TecplotWriter {
public:
    TecplotWriter(const Mesh& mesh, const std::string& filename)
        : mesh_(mesh), filename_(filename) {}
    
    // Add fields
    void addField(const ScalarField& field) { scalarFields_.push_back(&field); }
    void addField(const VectorField& field) { vectorFields_.push_back(&field); }
    
    // Write ASCII format
    void writeASCII();
    
    // Write binary format (PLT)
    void writeBinary();
    
private:
    const Mesh& mesh_;
    std::string filename_;
    std::vector<const ScalarField*> scalarFields_;
    std::vector<const VectorField*> vectorFields_;
    
    // Write zones
    void writeZone(std::ofstream& file);
    
    // Write connectivity
    void writeConnectivity(std::ofstream& file);
};

// Field probe writer
class ProbeWriter {
public:
    ProbeWriter(const std::string& filename)
        : filename_(filename) {
        file_.open(filename);
        writeHeader();
    }
    
    // Add probe location
    void addProbe(const Vector3& location, const std::string& name = "") {
        probes_.push_back({location, name.empty() ? 
                          "probe_" + std::to_string(probes_.size()) : name});
    }
    
    // Write probe data
    void write(Real time, const ScalarField& field);
    void write(Real time, const VectorField& field);
    
    // Flush data
    void flush() { file_.flush(); }
    
private:
    struct Probe {
        Vector3 location;
        std::string name;
    };
    
    std::string filename_;
    std::ofstream file_;
    std::vector<Probe> probes_;
    
    void writeHeader();
    
    // Interpolate field to probe location
    template<typename T>
    T interpolateToProbe(const Field<T>& field, const Vector3& location);
};

// Surface sampling writer
class SurfaceWriter {
public:
    SurfaceWriter(const Mesh& mesh, const std::string& surfaceName)
        : mesh_(mesh), surfaceName_(surfaceName) {}
    
    // Define surface by boundary patches
    void addPatch(const std::string& patchName) {
        patches_.push_back(patchName);
    }
    
    // Define surface by cutting plane
    void setPlane(const Vector3& point, const Vector3& normal) {
        planePoint_ = point;
        planeNormal_ = normal.normalized();
        isPlane_ = true;
    }
    
    // Write surface data
    void write(const std::string& filename, const ScalarField& field);
    void write(const std::string& filename, const VectorField& field);
    
private:
    const Mesh& mesh_;
    std::string surfaceName_;
    std::vector<std::string> patches_;
    Vector3 planePoint_;
    Vector3 planeNormal_;
    bool isPlane_ = false;
    
    // Extract surface mesh
    void extractSurface(std::vector<Vector3>& points,
                       std::vector<std::vector<int>>& faces);
    
    // Interpolate to surface
    template<typename T>
    void interpolateToSurface(const Field<T>& field,
                             std::vector<T>& surfaceData);
};

// Statistics writer
class StatisticsWriter {
public:
    StatisticsWriter(const std::string& filename)
        : filename_(filename) {}
    
    // Accumulate statistics
    void accumulate(const ScalarField& field);
    void accumulate(const VectorField& field);
    
    // Write statistics
    void write();
    
    // Reset statistics
    void reset();
    
private:
    std::string filename_;
    int samples_ = 0;
    
    // Statistics fields
    SharedPtr<ScalarField> mean_;
    SharedPtr<ScalarField> rms_;
    SharedPtr<ScalarField> min_;
    SharedPtr<ScalarField> max_;
    
    SharedPtr<VectorField> meanVector_;
    SharedPtr<TensorField> reynoldsStress_;
};

// Factory function
inline std::unique_ptr<VTKWriter> createWriter(
    const std::string& format,
    const Mesh& mesh,
    const std::string& baseName = "output") {
    
    if (format == "vtk" || format == "VTK") {
        auto writer = std::make_unique<VTKWriter>(mesh, baseName);
        writer->setFormat(VTKWriter::Format::BINARY);
        return writer;
    } else if (format == "vtu" || format == "VTU") {
        auto writer = std::make_unique<VTKWriter>(mesh, baseName);
        writer->setFormat(VTKWriter::Format::XML);
        return writer;
    } else {
        throw std::runtime_error("Unknown output format: " + format);
    }
}

} // namespace cfd::io