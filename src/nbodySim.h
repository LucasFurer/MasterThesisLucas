#ifndef NBODYSIM_H
#define NBODYSIM_H

#include "octtree.h"
#include "ffthelper.h"
#include "common.h"
#include <vector>

//float E = 2.71828182845904523536;

enum SimulationData {
    blueGreenCube,
    rainbowCube
};

class NbodySim
{
public:
    Particle3D* particles;
    std::size_t particlesSize;
    Buffer* particlesBuffer;

    std::vector<LineSegment> lineSegments;
    Buffer* boxBuffer = new Buffer();

    AccelerationType accelerationType;
    float t;
    float lastTimeSimulated;
    int simAmountPerSec;
    glm::vec3* acceleration;
    float gravConst;
    float softening;
    float theta;
    int meshAmount;

    int showLevel;

    NbodySim(Particle3D* initParticles, std::size_t initParticlesSize, AccelerationType initAccelerationType, float initT, int initSimAmountPerSec)
    {
        particlesSize = initParticlesSize;
        accelerationType = initAccelerationType;
        t = initT;
        lastTimeSimulated = 0.0f;
        simAmountPerSec = initSimAmountPerSec;

        int particleAmount = initParticlesSize / sizeof(Particle3D);

        acceleration = new glm::vec3[particleAmount];
        particles = new Particle3D[particleAmount];

        float* particlesToBuffer = Particle3D::Particle3DToFloat(particles, particlesSize);
        particlesBuffer = new Buffer(particlesToBuffer, 6 * sizeof(float) * (particlesSize/sizeof(Particle3D)), pos3DCol3D, GL_DYNAMIC_DRAW);
        delete[] particlesToBuffer;
    }

    NbodySim(SimulationData simulationData, AccelerationType initAccelerationType, float initT, int initSimAmountPerSec, int particleAmount, float initGravConst, float initSoftening, float initTheta, int initMeshAmount, int initShowLevel)
    {
        accelerationType = initAccelerationType;
        t = initT;
        lastTimeSimulated = 0.0f;
        simAmountPerSec = initSimAmountPerSec;
        gravConst = initGravConst;
        softening = initSoftening;
        theta = initTheta;
        showLevel = initShowLevel;

        switch (simulationData)
        {
        case blueGreenCube:
        {
            //int partSize = ParticleAmount;
            particlesSize = particleAmount * sizeof(Particle3D);
            
            particles = new Particle3D[particleAmount];
            acceleration = new glm::vec3[particleAmount];

            float sizeParam = 30.0f;
            for (int i = 0; i < particleAmount; i++)
            {
                glm::vec3 pos = glm::vec3(0.0f);
                glm::vec3 speed = glm::vec3(0.0f);
                glm::vec3 col = glm::vec3(0.0f);

                if (i % 2 == 0)
                {
                    pos = glm::vec3(
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f),
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f) - 25.0f,
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f)
                    );

                    speed = glm::vec3(
                        0.01f,
                        0.0f,
                        0.0f
                    );

                    col = glm::vec3(
                        0.0f,
                        0.0f,
                        1.0f
                    );
                }
                else
                {
                    pos = glm::vec3(
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f),
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f) + 25.0f,
                        powf(sizeParam * (float)rand() / RAND_MAX - (sizeParam / 2.0f), 1.0f)
                    );

                    speed = glm::vec3(
                        -0.01f,
                        0.0f,
                        0.0f
                    );

                    col = glm::vec3(
                        0.0f,
                        1.0f,
                        0.0f
                    );
                }


                particles[i] = Particle3D(pos, speed, col, 1.0f);
            }

            float* particlesToBuffer = Particle3D::Particle3DToFloat(particles, particlesSize);
            particlesBuffer = new Buffer(particlesToBuffer, 6 * sizeof(float) * (particlesSize / sizeof(Particle3D)), pos3DCol3D, GL_DYNAMIC_DRAW);
            delete[] particlesToBuffer;
        }
        break;
        case rainbowCube:
        {
            //int partSize = ParticleAmount;
            particlesSize = particleAmount * sizeof(Particle3D);

            particles = new Particle3D[particleAmount];
            acceleration = new glm::vec3[particleAmount];

            float sizeParam = 30.0f;
            for (int i = 0; i < particleAmount; i++)
            {
                glm::vec3 pos = glm::vec3(0.0f);
                glm::vec3 speed = glm::vec3(0.0f);
                glm::vec3 col = glm::vec3(0.0f);

                float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                float randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                while (powf(randX, 2.0f) + powf(randY, 2.0f) + powf(randZ, 2.0f) > 1.0f)
                {
                    randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                    randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                    randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                }

                pos = glm::vec3(
                    powf(sizeParam * randX, 1.0f),
                    powf(sizeParam * randY, 1.0f),
                    powf(sizeParam * randZ, 1.0f)
                );

                
                speed = glm::vec3(
                    (((float)rand() / RAND_MAX) - 0.5f) / 30.0f,
                    (((float)rand() / RAND_MAX) - 0.5f) / 30.0f,
                    (((float)rand() / RAND_MAX) - 0.5f) / 30.0f
                );
                

                col = glm::vec3(
                    randX,
                    1.0f,
                    1.0f
                );
                
                particles[i] = Particle3D(pos, speed, col, 1.0f);
            }
            float* particlesToBuffer = Particle3D::Particle3DToFloat(particles, particlesSize);
            particlesBuffer = new Buffer(particlesToBuffer, 6 * sizeof(float) * (particlesSize / sizeof(Particle3D)), pos3DCol3D, GL_DYNAMIC_DRAW);
            delete[] particlesToBuffer;
        }
        break;
        default:
            std::cout << "invalid SimulationData given" << std::endl;
        }
    }

    ~NbodySim()
    {
        delete[] particles;
        delete[] acceleration;
    }

    void simulate()
    {
        float currentTime = glfwGetTime();
        if (currentTime - lastTimeSimulated > 1.0f / (float)simAmountPerSec)
        {
            lastTimeSimulated = currentTime;

            switch (accelerationType) {
            case naive:
                naiveAcc();
                break;
            case barnesHut:
                barnesHutSimulate();
                break;
            case particleMesh:
                particleMeshAcc();
                break;
            default:
                std::cout << "invalid AccelerationType" << std::endl;
            }
            

            int particleAmount = particlesSize / sizeof(Particle3D);
            float maxDis = std::numeric_limits<float>::infinity();
            for (int i = 0; i < particleAmount; i++)
            {
                particles[i].speed += acceleration[i] * t;

                float expVar = 0.3f;
                float speedTemp = glm::length(particles[i].speed);
                float ratio = powf(2.0f / (1.0f + powf(E, -speedTemp)) - 1.0f, expVar);

                particles[i].color.r = ratio;//powf(2.0f/(1.0f+powf(E,-speedTemp))-1.0f, expVar);
                particles[i].color.g = 1.0f;//powf(2.0f/(1.0f+powf(E,-speedTemp))-1.0f, expVar);
                particles[i].color.b = 1.0f - ratio;


                if (std::abs(particles[i].position.x) > maxDis || std::abs(particles[i].position.y) > maxDis || std::abs(particles[i].position.z) > maxDis)
                {
                    particles[i].position -= particles[i].speed * t * 5.0f;
                    particles[i].speed = glm::vec3(0.0f);
                }
                else
                {
                    particles[i].position += particles[i].speed * t;
                }
            }

            float* particlesToBuffer = Particle3D::Particle3DToFloat(particles, particlesSize);
            particlesBuffer->updateBuffer(particlesToBuffer, 6 * sizeof(float) * (particlesSize / sizeof(Particle3D)), pos3DCol3D);
            delete[] particlesToBuffer;
        }
    }

    void particleMeshAcc()
    {
        //float* mesh = new float[];

        /*
        //code for in main
        const int N = 10;
        fftw_complex x[N];
        fftw_complex y[N];
        for (int i = 0; i < N; i++)
        {
            x[i][REAL] = i;
            x[i][IMAG] = N-i-1;
        }
        FFTHelper::fft(x, y, N);
        FFTHelper::displatComplex(x, N);
        FFTHelper::displatComplex(y, N);
        FFTHelper::ifft(y, x, N);
        FFTHelper::displatComplex(x, N);
        */
    }

    void naiveAcc()
    {
        int particleAmount = particlesSize / sizeof(Particle3D);
        glm::vec3 tempAcceleration;

        for (int i = 0; i < particleAmount; i++)
        {
            acceleration[i] = glm::vec3(0.0f, 0.0f, 0.0f);

            for (int j = 0; j < particleAmount; j++)
            {
                if (i != j)
                {
                    tempAcceleration = glm::normalize(particles[j].position - particles[i].position);
                    float Radius = glm::length(particles[j].position - particles[i].position);
                    tempAcceleration = (gravConst / (powf(Radius, 2.0f) + softening)) * tempAcceleration;

                    acceleration[i] += tempAcceleration;
                }
            }
        }
    }

    void barnesHutSimulate()
    {
        //lastTimeSimulated = currentTime;
        int particleAmount = particlesSize / sizeof(Particle3D);
        //glm::vec3 tempAcceleration;

        OctTree::maxChildren = 5;
        OctTree::allParticles = particles;
        OctTree::allParticlesSize = particlesSize;

        float keepTime = glfwGetTime();
        OctTree root = OctTree();
        //std::cout << "time for tree construction: " << glfwGetTime() - keepTime << std::endl;



        keepTime = glfwGetTime();
        for (int i = 0; i < particleAmount; i++)
        {
            acceleration[i] = getBarnesHutAcc(&root, particles[i]);
        }
        //std::cout << "time for tree traversal: " << glfwGetTime() - keepTime << std::endl;

        lineSegments.clear();
        root.getLineSegments(lineSegments, 0, showLevel);

        float* lineSegmentsToBuffer = LineSegment::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment));
        boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 12 * sizeof(float) * lineSegments.size(), pos3DCol3D, GL_DYNAMIC_DRAW);
        delete[] lineSegmentsToBuffer;
    }

    glm::vec3 getBarnesHutAcc(OctTree* node, Particle3D particle)
    {
        glm::vec3 acc(0.0f);

        float parCentreDistance = glm::length(node->centreOfMass - particle.position);
        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec3 cubeCentre = ((node->highestCorner + node->lowestCorner)/2.0f);
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            glm::vec3 forceDirection = glm::normalize(node->centreOfMass - particle.position);
            acc += ((gravConst * node->totalMass) / (powf(parCentreDistance, 2.0f) + softening)) * forceDirection;
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal(OctTree::allParticles[i].position, particle.position)))
                {
                    float parParDistance = glm::length(OctTree::allParticles[i].position - particle.position);

                    glm::vec3 forceDirection = glm::normalize(OctTree::allParticles[i].position - particle.position);
                    acc += ((gravConst * OctTree::allParticles[i].mass) / (powf(parParDistance, 2.0f) + softening)) * forceDirection;
                }
            }
        }
        else
        {
            for (OctTree* octTree : node->children)
            {
                acc += getBarnesHutAcc(octTree, particle);
            }
        }

        return acc;
    }

    void setAccelerationType(AccelerationType setAccelerationType)
    {
        accelerationType = setAccelerationType;
    }
private:
};


#endif
