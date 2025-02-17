#include "mpm_io.h"
#include <cstdio>

void SaveParticleDataToVTP(const char* filename,
                           const float* x, const float* y,
                           const float* vx, const float* vy,
                           int n)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file for write: %s\n", filename);
        return;
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PolyData>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfVerts=\"%d\">\n", n, n);

    // Points
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%f %f 0.0\n", x[i], y[i]);
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");

    // Verts connectivity
    fprintf(fp, "      <Verts>\n");
    fprintf(fp, "        <DataArray type=\"UInt32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d ", i);
    }
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "        <DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\">\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d ", i + 1);
    }
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "      </Verts>\n");

    // Velocity in PointData
    fprintf(fp, "      <PointData Scalars=\"Velocity\">\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%f %f 0.0\n", vx[i], vy[i]);
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </PointData>\n");

    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </PolyData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}
