#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// Needed on MsWindows
#define NOMINMAX
#include <windows.h>
#endif // Win32 platform

#include <math.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
// Download glut from: http://www.opengl.org/resources/libraries/glut/
#include <GLUT/glut.h>

#include "float2.h"
#include "float3.h"
#include "float4.h"
#include "float4x4.h"
#include <vector>

// simple material class, with object color, and headlight shading
class Material
{
protected:
    float3 color;
public:
    virtual float3 shade(float3 n, float3 v, float3 ld, float3 lp, float3 position)
    {
        return float3(0,0,0);
    }
    
    float snoise(float3 r) {
        unsigned int x = 0x0625DF73;
        unsigned int y = 0xD1B84B45;
        unsigned int z = 0x152AD8D0;
        float f = 0;
        for(int i=0; i<32; i++) {
            float3 s(	x/(float)0xffffffff,
                     y/(float)0xffffffff,
                     z/(float)0xffffffff);
            f += sin(s.dot(r));
            x = x << 1 | x >> 31;
            y = y << 1 | y >> 31;
            z = z << 1 | z >> 31;
        }
        return f / 64.0 + 0.5;
    }

    
    

};

class Diffuse : public Material
{
    float3 kd;
public:
    Diffuse(float3 k): kd(k){};
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 position)
    {
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        return kd * lightPowerDensity * cosTheta;
    }
};

class Wood : public Diffuse
{
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    Wood():
    Diffuse(float3(1, 1, 1))
    {
        scale = 16;
        turbulence = 500;
        period = 8;
        sharpness = 10;
    }
    float3 shade(float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 position)
    {
        
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
        w -= int(w);
        return (float3(1, 0.3, 0) * w + float3(0.35, 0.1, 0.05) * (1-w)) * lightPowerDensity * cosTheta;
    }
};

class WhiteMarble : public Diffuse
{
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    WhiteMarble():
    Diffuse(float3(1, 1, 1))
    {
        scale = 4;
        turbulence = 300;
        period = 10;
        sharpness = 6;
    }
    
    float3 shade(float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 position)
    {
        
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
        w = pow(sin(w)*0.5+0.5, 4);
        return (float3(0, 0, 0.3) * w + float3(0.97, 0.95, 0.99) * (1-w)) * lightPowerDensity * cosTheta;
    }
};

class BlackMarble : public Diffuse
{
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    BlackMarble():
    Diffuse(float3(1, 1, 1))
    {
        scale = 4;
        turbulence = 500;
        period = 10;
        sharpness = 6;
    }
    
    float3 shade(float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 position)
    {
        
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
        w = pow(sin(w)*0.5+0.5, 4);
        return (float3(0, 0, 0.9) * w + float3(0.02, 0.01, 0.05) * (1-w)) * lightPowerDensity * cosTheta;
    }
};

class PhongBlinn : public Material {
    float3 ks;
    float shininess;
    float3 kd;
public:
    PhongBlinn(float3 k, float s, float3 k2): ks(k), shininess(s), kd(k2) {};
    float3 shade( float3 normal, float3 viewDir,
                float3 lightDir, float3 lightPowerDensity, float3 position)
    {
        float cosTheta = normal.dot(lightDir);
        viewDir *= -1;
        float3 halfway =(viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        return lightPowerDensity * ks * pow(cosDelta, shininess) + (kd * lightPowerDensity * cosTheta);
    }
};

class PhongBlinnMarble : public Material {
    float3 ks;
    float shininess;
    float3 kd;
public:
    PhongBlinnMarble(float3 k, float s, float3 k2): ks(k), shininess(s), kd(k2) {};
    float3 shade( float3 normal, float3 viewDir,
                 float3 lightDir, float3 lightPowerDensity, float3 position)
    {
        float cosTheta = normal.dot(lightDir);
        viewDir *= -1;
        float3 halfway =(viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        
        float w = position.x * 10 + pow(snoise(position * 4), 6)*500;
        w = pow(sin(w)*0.5+0.5, 4);
        float3 x = ((float3(0, 0, 0.6) * w + float3(0.0, 0.0, 0) * (1-w)) * lightPowerDensity * cosTheta)*5;
        
        return x + (lightPowerDensity * ks * pow(cosDelta, shininess) + (kd * lightPowerDensity * cosTheta));
    }
};

class PhongBlinnNormal : public Material {
    float3 ks;
    float shininess;
    float3 kd;
public:
    PhongBlinnNormal(float3 k, float s, float3 k2): ks(k), shininess(s), kd(k2) {};
    float3 noise(float3 normal) {
        return float3(.6+.35*cos(10*normal.x), .6+.35*cos(10*normal.y), .6+.35*cos(10*normal.x));
    }
    float3 shade( float3 normal, float3 viewDir,
                 float3 lightDir, float3 lightPowerDensity, float3 position)
    {
        normal = noise (normal);
        float cosTheta = normal.dot(lightDir);
        viewDir *= -1;
        float3 halfway =(viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        
        float w = position.x * 10 + pow(snoise(position * 4), 6)*500;
        w = pow(sin(w)*0.5+0.5, 4);
        //float3 x = ((float3(0, 0, 0.6) * w + float3(0.0, 0.0, 0) * (1-w)) * lightPowerDensity * cosTheta)*5;
        float3 x (0,0,0);
        
        return x + (lightPowerDensity * ks * pow(cosDelta, shininess) + (kd * lightPowerDensity * cosTheta));
    }
};

class Metal : public Material {
    float3 r0;
public:
    Metal(float3  refractiveIndex, float3  extinctionCoefficient){
        float3 rim1 = refractiveIndex - float3(1,1,1);
        float3 rip1 = refractiveIndex + float3(1,1,1);
        float3 k2 = extinctionCoefficient * extinctionCoefficient;
        r0 = (rim1*rim1 + k2) / (rip1*rip1 + k2);
    }
    float3 shade( float3 normal, float3 viewDir,
                 float3 lightDir, float3 lightPowerDensity, float3 position)
    {
        float cosTheta = normal.dot(lightDir);
        viewDir *= -1;
        float3 halfway =(viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        return (lightPowerDensity * float3(0.1,0.1,1) * pow(cosDelta, 10) + (float3(3.13,2.23,1.76) * lightPowerDensity * cosTheta))*0.05;
    }
    struct Event{
        float3 reflectionDir;
        float3 reflectance;
    };
    Event evaluateEvent(float3 inDir, float3 normal) {
        Event e;
        float cosa = -normal.dot(inDir);
        float3 perp = -normal * cosa;
        float3 parallel = inDir - perp;
        e.reflectionDir = parallel - perp;
        e.reflectance = r0 + (float3(1,1,1)-r0) * pow(1 - cosa, 5);
        return e;
    }
};

class Gold : public Metal
{
public:
    Gold():Metal(float3 (0.21,0.485,1.29),float3 (3.13,2.23,1.76)){};
};

class Silver : public Metal
{
public:
    Silver():Metal(float3 (0.15,0.14,0.13),float3 (3.7,3.11,2.47)){};
};

class BlackMetal : public Metal
{
public:
    BlackMetal():Metal(float3 (0.1,0.1,0.1),float3 (0.1,0.1,0.1)){};
};

class Dielectric : public Material {
    float  refractiveIndex;
    float  r0;
public:
    Dielectric(float refractiveIndex): refractiveIndex(refractiveIndex) {
        r0 = (refractiveIndex - 1)*(refractiveIndex - 1)
        / (refractiveIndex + 1)*(refractiveIndex + 1);  }
    float3 shade( float3 normal, float3 viewDir,
                 float3 lightDir, float3 lightPowerDensity, float3 position)
    {
        return float3 (0.4,0.4,0.4);
    }
    
    struct Event{
        float3 reflectionDir;
        float3 refractionDir;
        float reflectance;
        float transmittance;
    };
    
    Event evaluateEvent(float3 inDir, float3& normal) {
        Event e;
        float cosa = -normal.dot(inDir);
        float3 perp = -normal * cosa;
        float3 parallel = inDir - perp;
        e.reflectionDir = parallel - perp;
        
        float ri = refractiveIndex;
        if (cosa < 0) { cosa = -cosa; normal = -normal; ri = 1/ri; }
        float disc = 1 - (1 - cosa * cosa) / ri / ri;
        if(disc < 0) e.reflectance = 1;
        else {
            float cosb = (disc < 0)?0:sqrt(disc);
            e.refractionDir = parallel / ri - normal * cosb;
            e.reflectance = r0 + (1 - r0) * pow(1 - cosa, 5);
        }
        e.transmittance = 1 - e.reflectance;
        return e;  }
};

class LightSource
{
public:
    virtual float3 getPowerDensityAt ( float3 x )=0;
    virtual float3 getLightDirAt     ( float3 x )=0;
    virtual float  getDistanceFrom   ( float3 x )=0;
};

class DirectionalLightSource : public LightSource
{
protected:
    float3 dir;
    float3 powerDensity;
public:
    DirectionalLightSource(float3 d, float3 p):dir(d), powerDensity(p){};
    float3 getPowerDensityAt(float3 x)
    {
        return powerDensity;
    }
    
    float3 getLightDirAt(float3 x)
    {
        return dir;
    }
    
    float getDistanceFrom (float3 x)
    {
        return 100000000;//FLT_MAX
    }
    
};

class PointLightSource : public LightSource
{
protected:
    float3 position;
    float3 power;
public:
    PointLightSource(float3 pos, float3 pow):position(pos), power(pow){};
    
    float getDistanceFrom(float3 x)
    {
        return sqrtf((position.x-x.x)*(position.x-x.x)+(position.y-x.y)*(position.y-x.y)+(position.z-x.z)*(position.z-x.z));
    }
    
    float3 getPowerDensityAt(float3 x)
    {
        float r = 1.0/(4.0*M_PI*getDistanceFrom(x)*getDistanceFrom(x));
        return power*r;
    }
    
    float3 getLightDirAt(float3 x)
    {
        return (position-x).normalize();
    }
};




// Skeletal camera class.
class Camera
{
	float3 eye;		//< world space camera position
	float3 lookAt;	//< center of window in world space
	float3 right;	//< vector from window center to window right-mid (in world space)
	float3 up;		//< vector from window center to window top-mid (in world space)

public:
	Camera()
	{
		eye = float3(0, 7, 25);//0, 0, 3
		lookAt = float3(0, 5, 19.5);// 0, 0, 2
		right = float3(1, 0, 0);//1 0 0
		up = float3(0, 1, 0);//0 1 0
	}
	float3 getEye()
	{
		return eye;
	}
	// compute ray through pixel at normalized device coordinates
	float3 rayDirFromNdc(const float2 ndc) {
		return (lookAt - eye
			+ right * ndc.x
			+ up    * ndc.y
			).normalize();
	}
};

// Ray structure.
class Ray
{
public:
    float3 origin;
    float3 dir;
    Ray(float3 o, float3 d)
    {
        origin = o;
        dir = d;
    }
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
	Hit()
	{
		t = -1;
	}
	float t;				//< Ray paramter at intersection. Negative means no valid intersection.
	float3 position;		//< Intersection coordinates.
	float3 normal;			//< Surface normal at intersection.
	Material* material;		//< Material of intersected surface.
};

// Object abstract base class.
class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material):material(material) {}
    virtual Hit intersect(const Ray& ray)=0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.
class QuadraticRoots
{
public:
	float t1;
	float t2;
	// Solves the quadratic a*a*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and set members t1 and t2 to store the roots.
	QuadraticRoots(float a, float b, float c)
	{
        float discr = b * b - 4.0 * a * c;
        if ( discr < 0 ) // no roots
		{
			t1 = -1;
			t2 = -1;
			return;
		}
        float sqrt_discr = sqrt( discr );
		t1 = (-b + sqrt_discr)/2.0/a;
		t2 = (-b - sqrt_discr)/2.0/a;
	}
	// Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
    float getLesserPositive() {
        return ((0 < t1 && t1 < t2) || t2 < 0)?t1:t2;
    }
};


class Plane : public Intersectable
{
protected:
    float3 normal;
    float3 r;
public:
    Plane(const float3& norm, float3 point, Material* material):
    Intersectable(material),
    normal(norm),
    r(point)
    {
    }
    
    float3 getNormalAt(float3 r)
    {
        return normal;
    }
    
Hit intersect(const Ray& ray)
{
    float t = ((r - ray.origin).dot(normal))/(ray.dir.dot(normal));
    Hit hit;
    hit.t = t;
    hit.position = ray.origin + ray.dir * t;
    hit.material = material;
    hit.normal = getNormalAt(hit.position);
    return hit;
    
}
};

class ChessBoard : public Plane
{
public:
    ChessBoard(const float3& norm, float3 point, Material* material):
    Plane(norm,point,material)
    {
    }
    
    Hit intersect(const Ray& ray)
    {
        float t = ((r - ray.origin).dot(normal))/(ray.dir.dot(normal));
        Hit hit;
        hit.t = t;
        hit.position = ray.origin + ray.dir * t;
        if (fabs(hit.position.x) <= 4.0 && fabs(hit.position.z) <= 4.0)
        {
            int posX = hit.position.x;
            int posZ = hit.position.z;
            if ((hit.position.x>=0 && hit.position.z>=0) || (hit.position.x<0 && hit.position.z<0))
            {
                if ((posX + posZ) %2 == 0) {
                    hit.material = new WhiteMarble();
                }
                else
                {
                    hit.material = new BlackMarble();
                }
            }
            else
            {
                if ((posX + posZ) %2 == 0) {
                    hit.material = new BlackMarble();
                }
                else
                {
                    hit.material = new WhiteMarble();
                }
            }
        }
        else
        {
            hit.material = new Diffuse(float3(1,1,1));
        }
        hit.normal = getNormalAt(hit.position);
        return hit;
        
    }
};

float4x4 sphereMatrix(float x)
{
    float4x4 matrix =  float4x4(1.0/x,0.0,0.0,0.0,
                                0.0,1.0/x,0.0,0.0,
                                0.0,0.0,1.0/x,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 ellipsoidMatrix(float x, float y, float z)
{
    float4x4 matrix =  float4x4(1.0/x,0.0,0.0,0.0,
                                0.0,1.0/y,0.0,0.0,
                                0.0,0.0,1.0/z,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}



float4x4 paraboloidMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,-1.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,0.0);
    
    return matrix;
}

float4x4 coneMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,-1.0,0.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,0.0);
    
    return matrix;
}

float4x4 upsideParaboloidMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,1.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,0.0);
    
    return matrix;
}

float4x4 hyperboloidMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,-1.0,0.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 queenBody()
{
    float4x4 matrix =  float4x4(8.0,0.0,0.0,0.0,
                                0.0,-1.0,5.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 knightBody()
{
    float4x4 matrix =  float4x4(4.0,-3.0,0.0,0.0,
                                0.0,-1.0,2.0,0.0,
                                0.0,0.2,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 hyperbolicParaboloidMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,-1.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 hyperbolicCylinderMatrix()
{
    float4x4 matrix =  float4x4(-1.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,1.0);
    
    return matrix;
}

float4x4 cylinderMatrix()
{
    float4x4 matrix =  float4x4(1.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

float4x4 horizontalCylinderMatrix()
{
    float4x4 matrix =  float4x4(0.0,0.0,0.0,0.0,
                                0.0,1.0,0.0,0.0,
                                0.0,0.0,1.0,0.0,
                                0.0,0.0,0.0,-1.0);
    
    return matrix;
}

class Quadric : public Intersectable
{
public:
    float4x4 coeffs;
    Quadric(Material* material):
    Intersectable(material)
    {
    }
    
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        float4 d = ray.dir;
        d.w = 0;
        float4 e = ray.origin;

        float a = d.dot(coeffs*d);
        float b = d.dot(coeffs*e)+e.dot(coeffs*d);
        float c = e.dot(coeffs*e);
        
        return QuadraticRoots(a, b, c);
    }
    
    float3 getNormalAt(float3 r)
    {
        float4 p = float4(r);
        p = operator*(p, coeffs) + coeffs*p;
        float3 result = float3(p.x,p.y,p.z);
        return result.normalize();
    }
    
    Hit intersect(const Ray& ray)
    {
        float t = solveQuadratic(ray).getLesserPositive();
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
        
    }
    
    
    Quadric* sphere(float x)
    {
        coeffs = sphereMatrix(x);
        return this;
    }
    
    Quadric* ellipsoid(float x, float y, float z)
    {
        coeffs = ellipsoidMatrix(x,y,z);
        return this;
    }
    
    
    Quadric* paraboloid()
    {
        coeffs = paraboloidMatrix();
        return this;
    }
    
    Quadric* upsideParaboloid()
    {
        coeffs = upsideParaboloidMatrix();
        return this;
    }
    
    Quadric* hyperboloid()
    {
        coeffs = hyperboloidMatrix();
        return this;
    }
    
    Quadric* queenBodyPiece()
    {
        coeffs = queenBody();
        return this;
    }
    
    Quadric* knightBodyPiece()
    {
        coeffs = knightBody();
        return this;
    }
    
    Quadric* hyperbolicParaboloid()
    {
        coeffs = hyperbolicParaboloidMatrix();
        return this;
    }
    
    Quadric* hyperbolicCylinder()
    {
        coeffs = hyperbolicCylinderMatrix();
        return this;
    }
    
    Quadric* cone()
    {
        coeffs = coneMatrix();
        return this;
    }
    
    Quadric* cylinder()
    {
        coeffs = cylinderMatrix();
        return this;
    }
    
    Quadric* horizontalCylinder()
    {
        coeffs = horizontalCylinderMatrix();
        return this;
    }
    
    Quadric* transform(float4x4 t)
    {
        float4x4 inv;
        inv = t.invert();
        coeffs = inv*coeffs*inv.transpose();
        return this;
    }
    
    bool contains(float3 r)
    {
        float4 rhomo(r);
        
        float something = rhomo.dot(coeffs*rhomo);
        
        if (something>0) return false;
        
        else return true;
    }
    
    Quadric* parallelPlanes(float height)
    {
        coeffs = float4x4::identity();
        coeffs._00 = 0;
        coeffs._11 = 1;
        coeffs._22 = 0;
        coeffs._33 = -1*(height/2)*(height/2);
        return this;
    }
    
    Quadric* parallelXPlanes(float height)
    {
        coeffs = float4x4::identity();
        coeffs._00 = 1;
        coeffs._11 = 0;
        coeffs._22 = 0;
        coeffs._33 = -1*(height/2)*(height/2);
        return this;
    }
};


class ClippedQuadric: public Intersectable {
    Quadric* shape;
    Quadric* clipper;

public:
    
    ClippedQuadric(Quadric* s, Quadric* c, Material* material):shape(s),clipper(c),Intersectable(material)
    {
    }
    
    Hit intersect(const Ray& ray)
    {
        QuadraticRoots roots = shape->solveQuadratic(ray);

        if (roots.t1>=0)
        {
            if (!clipper->contains(ray.origin + ray.dir * roots.t1))
            {
                roots.t1 = -1;
            }
        }
        if (roots.t2>=0)
        {
            if (!clipper->contains(ray.origin + ray.dir * roots.t2))
            {
                roots.t2 = -1;
            }
        }

        float t = roots.getLesserPositive();

        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = shape->getNormalAt(hit.position);
        return hit;
    }
    
    ClippedQuadric* transform(float4x4 t)
    {
        shape->transform(t);
        clipper->transform(t);
        return this;
    }
    
    ClippedQuadric* cylinder(float height)
    {
        shape->cylinder();
        clipper->parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* pawnCone(float height)
    {
        shape->cone()->transform(float4x4::translation(float3(0,0.5,0))*float4x4::scaling(float3(0.4, 0.5, 0.5)));
        clipper->parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* kingCone(float height)
    {
        shape->cone()->transform(float4x4::translation(float3(0,-0.5,0)));
        clipper->parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* crown(float height)
    {
        shape->cone();
        clipper->parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* halfSphere()
    {
        shape->sphere(1)->transform(float4x4::scaling(float3(1, 1, 1)) );
        clipper->sphere(1)->transform(float4x4::translation(float3(0.0, 0.5, 0.3)));
        return this;
    }
    ClippedQuadric* pringle()
    {
        shape->hyperboloid()->transform(float4x4::scaling(float3(1, 2, 1)) );
        clipper->sphere(1)->transform(float4x4::translation(float3(0.1, 0.0, 0.2)));
        return this;
    }
    
    ClippedQuadric* bishopBody()
    {
        shape->queenBodyPiece()->transform(float4x4::scaling(float3(1, 2, 1)) );
        clipper->parallelPlanes(1);
        return this;
    }
    
    ClippedQuadric* queenBody()
    {
        shape->queenBodyPiece()->transform(float4x4::scaling(float3(1, 2, 1)) );
        clipper->parallelPlanes(1.2);
        return this;
    }
    
    ClippedQuadric* knightBody()
    {
        shape->knightBodyPiece()->transform(float4x4::scaling(float3(1, 2, 1))*float4x4::translation(float3(0.0, 0.0, 0.0)));
        clipper->parallelPlanes(1);
        return this;
    }
    
    ClippedQuadric* kingParaboloid()
    {
        shape->upsideParaboloid();
        clipper->parallelPlanes(1);
        return this;
    }
};

class ClippedTriQuadric: public Intersectable {
    Quadric* shape;
    Quadric* clipperOne;
    Quadric* clipperTwo;
    Quadric* clipperThree;
    
public:
    
    ClippedTriQuadric(Quadric* s, Quadric* c1, Quadric* c2, Quadric* c3, Material* material):shape(s),clipperOne(c1),
                                                                                            clipperTwo(c2),clipperThree(c3),Intersectable(material)
    {
    }
    
    Hit intersect(const Ray& ray)
    {
        QuadraticRoots roots = shape->solveQuadratic(ray);
        
        if (roots.t1>=0)
        {
            if ((!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                !clipperTwo->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                !clipperTwo->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperTwo->contains(ray.origin + ray.dir * roots.t1) &&
                !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                !clipperOne->contains(ray.origin + ray.dir * roots.t1) ||
                !clipperTwo->contains(ray.origin + ray.dir * roots.t1) ||
                !clipperThree->contains(ray.origin + ray.dir * roots.t1))
            {
                roots.t1 = -1;
            }
        }
        if (roots.t2>=0)
        {
            if ((!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperTwo->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperTwo->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperTwo->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                (!clipperOne->contains(ray.origin + ray.dir * roots.t1) &&
                 !clipperThree->contains(ray.origin + ray.dir * roots.t1)) ||
                !clipperOne->contains(ray.origin + ray.dir * roots.t1) ||
                !clipperTwo->contains(ray.origin + ray.dir * roots.t1) ||
                !clipperThree->contains(ray.origin + ray.dir * roots.t1))
            {
                roots.t2 = -1;
            }
        }
        
        float t = roots.getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = shape->getNormalAt(hit.position);
        return hit;
    }
    
    ClippedTriQuadric* transform(float4x4 t)
    {
        shape->transform(t);
        clipperOne->transform(t);
        clipperTwo->transform(t);
        clipperThree->transform(t);
        return this;
    }
    
    ClippedTriQuadric* crown()
    {
        shape->cylinder()->transform(float4x4::translation(float3(0,0,0)));
        clipperOne->sphere(5)->transform(float4x4::translation(float3(-1,-0.5,0)));
        clipperTwo->sphere(5)->transform(float4x4::translation(float3(1,-0.5,0)));
        clipperThree->sphere(5)->transform(float4x4::translation(float3(0,2,0)));
        return this;
    }
    
    ClippedTriQuadric* kingCross()
    {
        shape->horizontalCylinder()->transform(float4x4::translation(float3(0,1,0)));
        clipperOne->ellipsoid(9, 1, 4)->transform(float4x4::translation(float3(0,1,0)));
        clipperTwo->ellipsoid(4, 9, 1)->transform(float4x4::translation(float3(0,1,0)));
        clipperThree->ellipsoid(1, 4, 9)->transform(float4x4::translation(float3(0,1,0)));
        return this;
    }
};



class Scene
{
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
    std::vector<LightSource*> lightSources;
public:
	Scene()
	{
        
        //LIGHTS
        lightSources.push_back(new DirectionalLightSource(float3(0,1,1),float3(1,1,1)));
//        lightSources.push_back(new DirectionalLightSource(float3(0,1,5),float3(1,1,1)));
        //lightSources.push_back(new DirectionalLightSource(float3(0,-1,0),float3(1,1,1)));
        //lightSources.push_back(new DirectionalLightSource(float3(0,0,1),float3(1,1,1)));
        //lightSources.push_back(new DirectionalLightSource(float3(0,0,-1),float3(1,1,1)));
        //lightSources.push_back(new DirectionalLightSource(float3(1,0,0),float3(1,1,1)));
        //lightSources.push_back(new DirectionalLightSource(float3(-1,0,0),float3(1,1,1)));
        lightSources.push_back(new PointLightSource(float3(.58,0,4.8),float3(30,30,30)));
        lightSources.push_back(new PointLightSource(float3(3,0,6),float3(100,100,100)));
        lightSources.push_back(new PointLightSource(float3(-3,0,6),float3(100,100,100)));
        lightSources.push_back(new PointLightSource(float3(0,0.5,1),float3(100,100,100)));
        
        //MATERIALS
        materials.push_back(new Gold());//0
        materials.push_back(new Silver());//1
        materials.push_back(new Diffuse(float3(1,0,0)));//2
        materials.push_back(new Wood());//3
        materials.push_back(new BlackMarble());//4
        materials.push_back(new WhiteMarble());//5
        materials.push_back(new Dielectric(1.46));//6
        materials.push_back(new PhongBlinn(float3(0,10,1),500,float3(1,0,10)));//7
        materials.push_back(new PhongBlinnMarble(float3(1,0,0),50,float3(1,0,1)));//8
        materials.push_back(new BlackMetal());//9
        materials.push_back(new PhongBlinnNormal(float3(0,10,1),50,float3(0,10,1)));//10
        materials.push_back(new PhongBlinn(float3(0,1,0),500,float3(0,1,0)));//11
        materials.push_back(new PhongBlinn(float3(0,0,1),500,float3(0,0.5,1)));//12
        
//        //PAWNS
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(-5.4,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                            pawnCone(1))->
                            transform(float4x4::translation(float3(-5.4,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(-3.85,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(-3.85,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(-2.3,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(-2.3,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(-0.7,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(-0.7,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(5.4,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(5.4,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(3.85,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(3.85,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(2.3,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(2.3,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(0.7,0.5,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(0.7,0,7.1))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        //Pawn in the background
        objects.push_back((new Quadric(materials[7]))->sphere(.1)->
                          transform(float4x4::translation(float3(5.4,0.5,-2))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[1]),new Quadric(materials[1]),materials[7]))->
                           pawnCone(1))->
                          transform(float4x4::translation(float3(5.4,0,-2))*float4x4::scaling(float3(0.6,0.6,0.6))));

//        //BISHOPS
        objects.push_back((new Quadric(materials[1]))->ellipsoid(0.2,0.4,1)
                          ->transform(float4x4::translation(float3(2.3,0.7,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[1]))->ellipsoid(0.4,0.001,1)
                          ->transform(float4x4::translation(float3(2.3,0.3,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[1]))->sphere(.02)
                          ->transform(float4x4::translation(float3(2.3,1.5,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[2]),new Quadric(materials[2]),materials[1]))->
                           bishopBody())->
                          transform(float4x4::translation(float3(2.3,0,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new Quadric(materials[1]))->ellipsoid(0.2,0.4,1)
                          ->transform(float4x4::translation(float3(-2.25,0.7,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[1]))->ellipsoid(0.4,0.001,1)
                          ->transform(float4x4::translation(float3(-2.25,0.3,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[1]))->sphere(.02)
                          ->transform(float4x4::translation(float3(-2.25,1.5,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[2]),new Quadric(materials[2]),materials[1]))->
                           bishopBody())->
                          transform(float4x4::translation(float3(-2.25,0,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
//
//        //KNIGHTS
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[12]))->
                           knightBody())->
                          transform(float4x4::translation(float3(3.8,0,8.9))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[12]))->
                          ellipsoid(0.7,0.1,0.7)
                          ->transform(float4x4::translation(float3(3.6,0.7,8.9))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[12]))->
                           knightBody())->
                          transform(float4x4::translation(float3(-3.8,0,8.9))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[12]))->
                          ellipsoid(0.7,0.1,0.7)
                          ->transform(float4x4::translation(float3(-3.5,0.7,8.9))*float4x4::scaling(float3(0.6,0.6,0.6))));
//
//        //ROOKS
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[11]))->
                           bishopBody())->
                          transform(float4x4::translation(float3(-5.3,0,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[11]))->
                           cylinder(1))->
                          transform(float4x4::translation(float3(-11.6,1.2,8.5))*float4x4::scaling(float3(0.27,0.4,0.6))));
        objects.push_back(((new Quadric(materials[11]))->ellipsoid(1, 0.01, 1))->
                          transform(float4x4::translation(float3(-11.6,1.7,8.5))*float4x4::scaling(float3(0.27,0.4,0.6))));
        
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[11]))->
                           bishopBody())->
                          transform(float4x4::translation(float3(5.3,0,8.5))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[11]))->
                           cylinder(1))->
                          transform(float4x4::translation(float3(11.6,1.2,8.5))*float4x4::scaling(float3(0.27,0.4,0.6))));
        objects.push_back(((new Quadric(materials[11]))->ellipsoid(1, 0.01, 1))->
                          transform(float4x4::translation(float3(11.6,1.7,8.5))*float4x4::scaling(float3(0.27,0.4,0.6))));
//
        //KING
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[6]))->
                           queenBody())->
                          transform(float4x4::translation(float3(0.8,0,8))*float4x4::scaling(float3(0.6,0.66,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[6]))->
                           kingCone(1))->
                          transform(float4x4::translation(float3(1,0.5,16))*float4x4::scaling(float3(.48,.72,.3))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[8]),new Quadric(materials[8]),materials[6]))->
                           kingParaboloid())->
                          transform(float4x4::translation(float3(1.18,2,8))*float4x4::scaling(float3(0.42,0.6,0.6))));
        objects.push_back((new ClippedTriQuadric(new Quadric(materials[2]),new Quadric(materials[8]),new Quadric(materials[8]),new Quadric(materials[8]),materials[6]))->kingCross()->
                          transform(float4x4::translation(float3(2.8,6.6,26.3333))*float4x4::scaling(float3(0.18,0.18,0.18))));
        objects.push_back((new Quadric(materials[6]))->ellipsoid(0.7,0.15,0.5)
                          ->transform(float4x4::translation(float3(0.8,1.3,8))*float4x4::scaling(float3(0.6,0.6,0.6))));

//        //QUEEN
        objects.push_back((new Quadric(materials[0]))->sphere(.05)
                          ->transform(float4x4::translation(float3(-0.75,1.85,8.2))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back((new Quadric(materials[0]))->ellipsoid(0.45,0.3,0.5)
                          ->transform(float4x4::translation(float3(-0.8,0.6,8.2))*float4x4::scaling(float3(0.6,0.6,0.6))));
        objects.push_back(((new ClippedQuadric(new Quadric(materials[0]),new Quadric(materials[0]),materials[0]))->queenBody())
                          ->transform(float4x4::translation(float3(-0.8,0,8.2))*float4x4::scaling(float3(0.6,0.6,0.6))));
        
        objects.push_back((new ClippedTriQuadric(new Quadric(materials[2]),new Quadric(materials[2]),new Quadric(materials[2]),new Quadric(materials[2]),materials[0]))->crown()
                          ->transform(float4x4::translation(float3(-1.2,0.8,7.9))*float4x4::scaling(float3(0.4,0.4,0.6))*float4x4::rotation(float3(0,0,0),0)));
        
        

        objects.push_back(new ChessBoard(float3(0, 1, 0), float3(0,-1,0), materials[0]));

	}
	~Scene()
	{
		// UNCOMMENT THESE WHEN APPROPRIATE
//		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
//			delete *iMaterial;
//		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
//			delete *iObject;		
	}

public:
    Hit firstIntersect(Ray ray) {
        
        Hit firstHit;
        float t = 100000000;
        for(int i = 0; i<objects.size(); i++)
        {
            Hit h = objects.at(i)->intersect(ray);
            if(h.t < t && h.t>0)
            {
                t = h.t;
                firstHit = h;
            }
        }
        
        if (t==100000000)
            firstHit.t=-1;
        
        return firstHit;
    }
    
	Camera& getCamera()
	{
		return camera;
	}

	float3 trace(const Ray& ray,int depth, float3 s, float3 q)
	{
        if(depth>3) return float3(1, 1, 1);
        
        Hit hit = firstIntersect(ray);
        // hit provides x, n, material

		if(hit.t < 0)
			return float3(1, 1, 1);
        
        float3 outRadiance = float3(0, 0, 0);
        
        for (int i = 0; i<lightSources.size(); i++)
        {
            Ray shadowRay(hit.position+ hit.normal*0.01, lightSources.at(i)->getLightDirAt(hit.position));
            
            Hit shadowHit = firstIntersect(shadowRay);
            
            if(shadowHit.t > 0 && shadowHit.t < lightSources.at(i)->getDistanceFrom(hit.position)) continue;
            
            outRadiance += hit.material->shade(hit.normal, ray.dir, lightSources.at(i)->getLightDirAt(hit.position),lightSources.at(i)->getPowerDensityAt(hit.position), hit.position);
            
            
        }
        
        Metal* metal = dynamic_cast<Metal*>(hit.material);
        if(metal != NULL){
            Metal::Event e = metal->evaluateEvent(ray.dir, hit.normal);
            outRadiance += trace( Ray(hit.position + hit.normal*0.001, e.reflectionDir),depth+1,s,q) * e.reflectance;
        }
        
        Dielectric* dielectric=dynamic_cast<Dielectric*>(hit.material);
        if(dielectric != NULL) {
            Dielectric::Event e = dielectric->evaluateEvent(ray.dir, hit.normal);
            outRadiance += trace( Ray(hit.position + hit.normal*0.001, e.reflectionDir),depth+1,s,q) * e.reflectance;
            if(e.transmittance > 0)
                outRadiance += trace( Ray(hit.position - hit.normal*0.001, e.refractionDir),depth+1,s,q) * e.transmittance;
            
            outRadiance.x *= exp(-s.x*hit.t);
            outRadiance.y *= exp(-s.y*hit.t);
            outRadiance.z *= exp(-s.z*hit.t);
            
            outRadiance.x += q.x * (1-exp(-s.x*hit.t))/s.x;
            outRadiance.y += q.y * (1-exp(-s.y*hit.t))/s.y;
            outRadiance.z += q.z * (1-exp(-s.z*hit.t))/s.z;
            
        }
    
        return outRadiance;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
// global application data

// screen resolution
const int screenWidth = 600;
const int screenHeight = 600;
// image to be computed by ray tracing
float3 image[screenWidth*screenHeight];

Scene scene;

bool computeImage()
{
	static unsigned int iPart = 0;

	if(iPart >= 64)
		return false;
    for(int j = iPart; j < screenHeight; j+=64)
	{
        for(int i = 0; i < screenWidth; i++)
		{
			float3 pixelColor = float3(0, 0, 0);
			float2 ndcPixelCentre( (2.0 * i - screenWidth) / screenWidth, (2.0 * j - screenHeight) / screenHeight );

			Camera& camera = scene.getCamera();
			Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcPixelCentre));
			
			image[j*screenWidth + i] = scene.trace(ray,1, float3(0.05,0.05,0.05), float3(0,0,0));
		}
	}
	iPart++;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL starts here. In the ray tracing example, OpenGL just outputs the image computed to the array.

// display callback invoked when window needs to be redrawn
void onDisplay( ) {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

	if(computeImage())
		glutPostRedisplay();
    glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
 
    glutSwapBuffers(); // drawing finished
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);						// initialize GLUT
    glutInitWindowSize(screenWidth, screenHeight);				// startup window size 
    glutInitWindowPosition(100, 100);           // where to put window on screen
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);    // 8 bit R,G,B,A + double buffer + depth buffer
 
    glutCreateWindow("Ray caster");				// application window is created and displayed
 
    glViewport(0, 0, screenWidth, screenHeight);

    glutDisplayFunc(onDisplay);					// register callback
 
    glutMainLoop();								// launch event handling loop
    
    return 0;
}

