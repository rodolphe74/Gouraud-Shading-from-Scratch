import java.util.Arrays;



class Light {
    float x, y, z;
    color c;
    float intensity;    // 0->255

    public Light(float x, float y, float z, color c, float intensity) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.c = c;
        this.intensity = intensity;
    }

    PVector getLightDirAsVector() {
        return new PVector(x, y, z);
    }
}

class Vertex {
    float x, y, z;
    float nx, ny, nz;

    Vertex(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    PVector getAsVector() {
        return new PVector(x, y, z);
    }

    PVector getNormalAsVector() {
        return new PVector(nx, ny, nz);
    }

    float[] getAsFloatArray() {
        float[] f = new float[4];
        f[0] = x;
        f[1] = y;
        f[2] = z;
        f[3] = 1;
        return f;
    }
}

class Face {
    int[] indices;
    int[] normals;
    int numberOfVertices;

    Face(int[] indices, int[] normals) {
        this.indices = indices;
        this.normals = normals;
        numberOfVertices = indices.length;
    }
}

class GrVertex {
    float x, y;
    float ooz;
    color c;

    GrVertex(float x, float y, color c) {
        this.x = x;
        this.y = y;
        this.c = c;
    }

    GrVertex(float x, float y) {
        this.x = x;
        this.y = y;
        this.c = color(80, 80, 80);
    }

    public String toString() {
        return "(" + x + "," + y + ")";
    }
}

class Edge {
    float x1, y1;
    float x2, y2;
    float z1, z2;
    color c1, c2;

    Edge(float x1, float y1, float x2, float y2, float z1, float z2, color c1, color c2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.z1 = z1;
        this.z2 = z2;
        this.c1 = c1;
        this.c2 = c2;
    }

    public String toString() {
        return "(" + x1 + "," + y1 + "-" + x2 + "," + y2 + ") ->" + red(c1) + "," + green(c1) + "," + blue(c1)
            + " - " + red(c2) + "," + green(c2) + "," + blue(c2);
    }
}

final int W = 640;
final int H = 640;

// Gouraud/Specular parameters
// http://devernay.free.fr/cours/opengl/materials.html

// Bronze
float[] diffuseLightColor = {  1.0f, 0.5f, 0.31f }; // white light diffuse
PVector specularLightColor = new PVector(0.5f, 0.5f, 0.5f); // yellow light speculare
float[] ambient = { 1.0f, 0.5f, 0.31f };
float specularStrength = 1.0f;
int shininess = 52;

// Turquoise
//    float[] diffuseLightColor = {  0.396f, 0.74151f, 0.69102f };
//    PVector specularLightColor = new PVector( 0.297254f, 0.30829f, 0.306678f);
//    float[] ambient = { 0.1f, 0.18725f,    0.1745f };
//    float specularStrength = 1.0f;
//    int shininess = 26;

// Black rubber
//    float[] diffuseLightColor = { 0.01f, 0.01f,    0.01f };
//    PVector specularLightColor = new PVector( 0.4f,     0.4f,     0.4f);
//    float[] ambient = { 0.02f, 0.02f, 0.02f };
//    float specularStrength = 1.0f;
//    int shininess = 20;

// Chrome
//float[] diffuseLightColor = { 0.4f, 0.4f, 0.4f };
//PVector specularLightColor = new PVector( 0.774597f, 0.774597f, 0.774597f);
//float[] ambient = { 0.25f, 0.25f, 0.25f };
//float specularStrength = 1.0f;
//int shininess = 150;

// Projection matrix
PMatrix3D viewPM = new PMatrix3D();
PMatrix3D perspectivePM = new PMatrix3D();
PVector fromPosition;
PVector toTarget;
PVector up;

// 3d Object
ArrayList<Vertex> vertices;
ArrayList<Face> faces;
ArrayList<PVector> normals;
BufferedReader reader;
String line;
Light light;
PVector objectColor = new PVector(1.0f * 255, 1.0f * 255, 1.0f * 255);

// z-buffer
float[] zBuffer = new float[W * H];



void addVertex(String[] parts) {
    Vertex vertex = new Vertex(Float.parseFloat(parts[1]), Float.parseFloat(parts[2]), Float.parseFloat(parts[3]));
    vertices.add(vertex);
}

void addNormal(String[] parts) {
    PVector normal = new PVector(Float.parseFloat(parts[1]), Float.parseFloat(parts[2]), Float.parseFloat(parts[3]));
    normals.add(normal);
}

void addFace(String[] parts) {
    int numberOfVertices = parts.length - 1;
    String part;
    String[] subParts;
    int[] indices = new int[numberOfVertices];
    int[] normals = new int[numberOfVertices];
    for (int i = 0; i < numberOfVertices; i++) {
        part = parts[i + 1];
        subParts = splitTokens(part, "/");
        indices[i] = Integer.parseInt(subParts[0]) - 1;
        normals[i] = Integer.parseInt(subParts[2]) - 1;
    }
    Face face = new Face(indices, normals);
    faces.add(face);
}

void processLine(String line) {
    String[] parts = splitTokens(line);
    if (parts[0].equals("v")) {
        addVertex(parts);
    } else if (parts[0].equals("vn")) {
        addNormal(parts);
    } else if (parts[0].equals("f")) {
        addFace(parts);
    } else {
        // doNothing
    }
}

void readFile(String file) {
    vertices = new ArrayList<Vertex>();
    normals = new ArrayList<PVector>();
    faces = new ArrayList<Face>();
    reader = createReader(file);
    do {
        try {
            line = reader.readLine();
        }
        catch (IOException e) {
            e.printStackTrace();
            line = null;
        }
        if (line == null) {
            return;
        } else {
            processLine(line);
        }
    } while (true);
}


float[] vec4MultiplyMat4(float[] result, float[] v0, PMatrix3D m0)
{
    float x = v0[0];
    float y = v0[1];
    float z = v0[2];
    float w = v0[3];

    result[0] = m0.m00 * x + m0.m10 * y + m0.m20 * z + m0.m30 * w;
    result[1] = m0.m01 * x + m0.m11 * y + m0.m21 * z + m0.m31 * w;
    result[2] = m0.m02 * x + m0.m12 * y + m0.m22 * z + m0.m32 * w;
    result[3] = m0.m03 * x + m0.m13 * y + m0.m23 * z + m0.m33 * w;
    return result;
}

float[] vec4Divide(float[] result, float[] v0, float f)
{
    result[0] = v0[0] / f;
    result[1] = v0[1] / f;
    result[2] = v0[2] / f;
    result[3] = v0[3] / f;
    return result;
}

float[] vec3Multiply(float[] result, float[] v0, float[] v1)
{
    result[0] = v0[0] * v1[0];
    result[1] = v0[1] * v1[1];
    result[2] = v0[2] * v1[2];
    return result;
}

float[] vec3Add(float[] result, float[] v0, float[] v1)
{
    result[0] = v0[0] + v1[0];
    result[1] = v0[1] + v1[1];
    result[2] = v0[2] + v1[2];
    return result;
}

PVector reflect(PVector out, PVector incident, PVector normal)
{
    float dot = incident.dot(normal);
    normal = PVector.mult(normal, dot);
    normal = PVector.mult(normal, 2);
    out = PVector.sub(incident, normal);
    out = out.normalize();
    return out;
}


void lookAt(PVector from, PVector to, PVector up, PMatrix3D mat) {
    // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    PVector forward = PVector.sub(from, to);
    forward.normalize();
    PVector right = up.cross(forward);
    right.normalize();
    PVector newup = forward.cross(right);

    mat.set(
        right.x, right.y, right.z, 0.0f,
        newup.x, newup.y, newup.z, 0.0f,
        forward.x, forward.y, forward.z, 0.0f,
        from.x, from.y, from.z, 1.0f
        );
}

void lookAt2(PVector position, PVector target, PVector up, PMatrix3D mat)
{
    // https://github.com/felselva/mathc
    PVector forward = PVector.sub(target, position);
    forward.normalize();
    PVector side = forward.cross(up);
    side.normalize();

    mat.set(
        side.x, up.x, -forward.x, 0.0f,
        side.y, up.y, -forward.y, 0.0f,
        side.z, up.z, -forward.z, 0.0f,
        -side.dot(position), -up.dot(position), forward.dot(position), 1.0f
        );
}

void perspective(float angleOfView, float near, float far, PMatrix3D mat) {
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
    float scale = (float) (1.0f / Math.tan(angleOfView * 0.5f * Math.PI / 180.0f));
    mat.set(scale, 0.0f, 0.0f, 0.0f,
        0.0f, scale, 0.0f, 0.0f,
        0.0f, 0.0f, -far / (far - near), -1.0f,
        0.0f, 0.0f, -far * near / (far - near), 0.0f
        );
}



void perspective2(float fov_y, float aspect, float n, float f, PMatrix3D mat) {
    // https://github.com/felselva/mathc
    float tan_half_fov_y = (float) (1.0f / Math.tan(fov_y * 0.5f));
    mat.set(1.0f / aspect * tan_half_fov_y, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tan_half_fov_y, 0.0f, 0.0f,
        0.0f, 0.0f, f / (n - f), -1.0f,
        0.0f, 0.0f, -(f * n) / (f - n), 0.0f
        );
}


void transformObject(PMatrix3D mat) {
    float[] result = new float[4];
    float[] normal = new float[4];

    for (Vertex v : vertices) {
        vec4MultiplyMat4(result, v.getAsFloatArray(), mat);
        v.x = result[0];
        v.y = result[1];
        v.z = result[2];
    }

    for (PVector n : normals) {
        n.get(normal);
        vec4MultiplyMat4(result, normal, mat);
        n.set(result[0], result[1], result[2]);
    }
}


boolean pnpoly(GrVertex[] p, GrVertex t) {
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    int i, j = 0;
    boolean c = false;
    for (i = 0, j = p.length - 1; i < p.length; j = i++) {
        if (((p[i].y > t.y) != (p[j].y > t.y))
            && (t.x < (p[j].x - p[i].x) * (t.y - p[i].y) / (p[j].y - p[i].y) + p[i].x))
            c = !c;
    }
    return c;
}

GrVertex[] findSquare(GrVertex[] polygon, GrVertex[] square) {
    square = new GrVertex[4];

    float minx = Float.MAX_VALUE, miny = Float.MAX_VALUE, maxx = Float.MIN_VALUE, maxy = Float.MIN_VALUE;
    for (GrVertex p : polygon) {
        if (p.x < minx)
            minx = p.x;
        if (p.x > maxx)
            maxx = p.x;
        if (p.y < miny)
            miny = p.y;
        if (p.y > maxy)
            maxy = p.y;
    }
    square[0] = new GrVertex(minx, miny);
    square[1] = new GrVertex(maxx, miny);
    square[2] = new GrVertex(maxx, maxy);
    square[3] = new GrVertex(minx, maxy);

    // System.out.println("Square:" + square[0] + "-" + square[1] + "-" + square[2]
    // + "-" + square[3]);
    return square;
}

Edge findEdge(int x, int y, Edge[] edges) {
    float dist1, dist2, dist3, diff, diffMin = Float.MAX_VALUE;
    int idx = -1;
    for (int i = 0; i < edges.length; i++) {

        if (edges[i].x2 - edges[i].x1 != 0) {
            // remove horizontal lines
            float slope = ((float)  edges[i].y2 - edges[i].y1) / ((float) edges[i].x2 - edges[i].x1);
            if (slope == 0.0f) {
                continue;
            }
        }

        dist1 = (float) Math.sqrt(Math.pow(x - edges[i].x1, 2) + Math.pow(y - edges[i].y1, 2));
        dist2 = (float) Math.sqrt(Math.pow(x - edges[i].x2, 2) + Math.pow(y - edges[i].y2, 2));
        dist3 = (float) Math.sqrt(Math.pow(edges[i].x2 - edges[i].x1, 2) + Math.pow(edges[i].y2 - edges[i].y1, 2));
        diff = abs((dist1 + dist2) - dist3);
        if (diff < diffMin) {
            diffMin = diff;
            idx = i;
        }
    }
    return edges[idx == -1 ? 0 : idx];
}

Edge[] returnEdges(GrVertex[] vertices, Edge[] edges) {
    edges = new Edge[vertices.length];
    int count = 0;
    for (int i = 0; i < vertices.length; i++) {
        Edge e = new Edge(vertices[i].x, vertices[i].y, vertices[(i + 1) % vertices.length].x,
            vertices[(i + 1) % vertices.length].y, vertices[i].ooz, vertices[(i + 1) % vertices.length].ooz,
            vertices[i].c, vertices[(i + 1) % vertices.length].c);
        edges[count] = e;
        count++;
    }
    return edges;
}

void doPoly(GrVertex[] p) {
    GrVertex[] square = null;
    square = findSquare(p, square);
    Edge[] edges = null;
    edges = returnEdges(p, edges);
    int x1, x2;
    int sum = 0;
    float r5, g5, b5, r6, g6, b6;
    float r, g, b;
    float z1, z2, z;

    for (int y = Math.round(square[0].y); y <= square[2].y; y++) {
        x1 = -1;
        x2 = -1;
        sum = 0;

        for (int x = Math.round(square[0].x); x <= square[1].x; x++) {
            if (pnpoly(p, new GrVertex(x, y)) == true) {
                if (x1 < 0)
                    x1 = x;
            } else {
                if (x1 >= 0) {
                    x2 = x;
                    sum += 1;
                }
            }
            if (sum >= 0 && sum % 2 != 0) {
                break;
            }
        }

        Edge leftEdge = findEdge(x1, y, edges);
        Edge rightEdge = findEdge(x2, y, edges);

        // z-buffer
        z1 = (((float) y - leftEdge.y2) / (leftEdge.y1 - leftEdge.y2)) * leftEdge.z1
            + (((float) leftEdge.y1 - y) / (leftEdge.y1 - leftEdge.y2)) * leftEdge.z2;

        z2 = (((float) y - rightEdge.y2) / (rightEdge.y1 - rightEdge.y2)) * rightEdge.z1
            + (((float) rightEdge.y1 - y) / (rightEdge.y1 - rightEdge.y2)) * rightEdge.z2;


        // Gouraud
        r5 = (((float) y - leftEdge.y2) / (leftEdge.y1 - leftEdge.y2)) * red(leftEdge.c1)
            + (((float) leftEdge.y1 - y) / (leftEdge.y1 - leftEdge.y2)) * red(leftEdge.c2);
        g5 = (((float) y - leftEdge.y2) / (leftEdge.y1 - leftEdge.y2)) * green(leftEdge.c1)
            + (((float) leftEdge.y1 - y) / (leftEdge.y1 - leftEdge.y2)) * green(leftEdge.c2);
        b5 = (((float) y - leftEdge.y2) / (leftEdge.y1 - leftEdge.y2)) * blue(leftEdge.c1)
            + (((float) leftEdge.y1 - y) / (leftEdge.y1 - leftEdge.y2)) * blue(leftEdge.c2);

        r6 = (((float) y - rightEdge.y2) / (rightEdge.y1 - rightEdge.y2)) * red(rightEdge.c1)
            + (((float) rightEdge.y1 - y) / (rightEdge.y1 - rightEdge.y2)) * red(rightEdge.c2);
        g6 = (((float) y - rightEdge.y2) / (rightEdge.y1 - rightEdge.y2)) * green(rightEdge.c1)
            + (((float) rightEdge.y1 - y) / (rightEdge.y1 - rightEdge.y2)) * green(rightEdge.c2);
        b6 = (((float) y - rightEdge.y2) / (rightEdge.y1 - rightEdge.y2)) * blue(rightEdge.c1)
            + (((float) rightEdge.y1 - y) / (rightEdge.y1 - rightEdge.y2)) * blue(rightEdge.c2);

        for (int x = x1; x < x2; x++) {
            if (x1 == x2) {
                z = z1;
                r = r5;
                g = g5;
                b = b5;
            } else {
                z = ((float) (x2 - x) / (x2 - x1)) * z1
                    + ((float) (x - x1) / (x2 - x1)) * z2;
                r = ((float) (x2 - x) / (x2 - x1)) * r5 + ((float) (x - x1) / (x2 - x1)) * r6;
                g = ((float) (x2 - x) / (x2 - x1)) * g5 + ((float) (x - x1) / (x2 - x1)) * g6;
                b = ((float) (x2 - x) / (x2 - x1)) * b5 + ((float) (x - x1) / (x2 - x1)) * b6;
            }


            int offset = x + y * W;
            if (offset >= 0 && offset < W * H && z > zBuffer[x + y * W]) {
                pixels[x + W * y] = color(r, g, b);
                zBuffer[x + W * y] = z;
            }
        }
    }
}

void render() {

    Arrays.fill(zBuffer, -Float.MAX_VALUE);

    for (Face f : faces) {
        GrVertex[] g = new GrVertex[f.numberOfVertices];
        for (int j = 0; j < f.numberOfVertices; j++) {
            Vertex v = vertices.get(f.indices[j]);
            PVector worldPos = v.getAsVector();
            PVector worldNorm = normals.get(f.normals[j]);

            // Gouraud ////////////
            worldNorm = worldNorm.normalize();
            PVector lightPos = light.getLightDirAsVector();
            PVector lightDir = lightPos.sub(worldPos);
            lightDir.normalize();
            float diff = Math.max(worldNorm.dot(lightDir), 0);
            PVector _diffuseLightColor = new PVector(diffuseLightColor[0], diffuseLightColor[1], diffuseLightColor[2]);
            PVector diffuse = _diffuseLightColor.mult(diff);
            PVector _ambient = new PVector(ambient[0], ambient[1], ambient[2]);
            PVector ambientDiffuseSpecular = _ambient.add(diffuse);

            // Specular ///////////
            PVector viewDir = fromPosition.copy();
            viewDir.sub(worldPos);
            viewDir.normalize();
            PVector negLightDir = new PVector(-lightDir.x, -lightDir.y, -lightDir.z);
            PVector reflectDir = new PVector();
            reflectDir = reflect(reflectDir, /*lightDir */  negLightDir, worldNorm);
            float spec = (float) Math.pow((float) Math.max(viewDir.dot(reflectDir), 0.0f), (float) shininess);
            PVector specular = new PVector();
            specular = PVector.mult(specularLightColor, spec);
            specular = PVector.mult(specular, specularStrength);

            float[] lightColorVec = { red(light.c), green(light.c), blue(light.c) };
            float[] specularVec = { specular.x, specular.y, specular.z };
            vec3Multiply(specularVec, specularVec, lightColorVec);
            // System.out.println(specularVec[0] + " | " + specularVec[1] + " | " + specularVec[2]);


            // Melt lights with object color
            float[] objectColorVec = new float[3];
            objectColor.get(objectColorVec);
            float[] c = new float[3];
            float[] ambientDiffuseSpecularVec = { ambientDiffuseSpecular.x, ambientDiffuseSpecular.y, ambientDiffuseSpecular.z };
            vec3Add(ambientDiffuseSpecularVec, ambientDiffuseSpecularVec, specularVec);
            vec3Multiply(c, ambientDiffuseSpecularVec, objectColorVec);


            // Projection /////////
            float[] worldPosVec = { worldPos.x, worldPos.y, worldPos.z, 1 };
            float[] cameraPosVec = new float[4];
            vec4MultiplyMat4(cameraPosVec, worldPosVec, viewPM);
            float[] projectionPosVec = new float[4];
            vec4MultiplyMat4(projectionPosVec, cameraPosVec, perspectivePM);
            vec4Divide(projectionPosVec, projectionPosVec, projectionPosVec[3]);
            for (int i = 0; i < projectionPosVec.length; i++)
                projectionPosVec[i] /= projectionPosVec[3];
            g[j] = new GrVertex(Math.min(W - 1, Math.round((projectionPosVec[0] + 1) * 0.5 * W)),
                Math.min(H - 1, Math.round((1 - (projectionPosVec[1] + 1) * 0.5) * H)));
            g[j].ooz = cameraPosVec[2];
            g[j].c = color(
                Math.min(255, Math.max(0, (int) Math.round(c[0]))),
                Math.min(255, Math.max(0, (int) Math.round(c[1]))),
                Math.min(255, Math.max(0, (int) Math.round(c[2]))));
        }

        // Render vertex here
        doPoly(g);

        //            en fil de fer (loadPixels & updatePixels as comments in draw())
        //            stroke(64);
        //            smooth();
        //            for (int i = 0; i < g.length; i++) {
        //                line(g[i].x, g[i].y, g[(i + 1) % g.length].x, g[(i + 1) % g.length].y);
        //            }
    }
}

public void settings() {
    size(W, H);
    readFile("loupiotte.obj");
    println("Read " + faces.size() + " faces");
    println("Read " + vertices.size() + " vertices");

    fromPosition = new PVector(0.0f, 0.0f, 5.0f);
    toTarget = new PVector(0.0f, 0.0f, 0.0f);
    up = new PVector(0.0f, 1.0f, 0.0f);
    lookAt2(fromPosition, toTarget, up, viewPM);
    viewPM.print();
    //        perspective(90.0f, 0.1f, 100.0f, perspectivePM);
    perspective2((float) Math.toRadians(90.0f), 1.0f, 0.1f, 100.0f, perspectivePM);
    viewPM.print();
    perspectivePM.print();

    light = new Light(0.0f, 0.0f, 8.0f, color(255, 255, 255), 255.0f);

    PMatrix3D rotationMatX = new PMatrix3D();
    rotationMatX.rotateX((float) Math.toRadians(0.0f));
    transformObject(rotationMatX);

    //        PMatrix3D translateMat = new PMatrix3D();
    //        translateMat.translate(100.0f, -40.0f);
    //        transformObject(translateMat);
}

public void draw() {
    background(210);
    loadPixels();
    PMatrix3D rotationMatY = new PMatrix3D();
    PMatrix3D rotationMatZ = new PMatrix3D();
    rotationMatY.rotateY((float) Math.toRadians(1.0f));
    transformObject(rotationMatY);
    rotationMatZ.rotateZ((float) Math.toRadians(0.8f));
    transformObject(rotationMatZ);

    //        long start = System.currentTimeMillis();
    render();
    //        long end = System.currentTimeMillis();
    //        System.out.println("Time:" + (end - start) + " millisecs");
    updatePixels();
}
