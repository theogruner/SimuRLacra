/*******************************************************************************
 Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
 Technical University of Darmstadt.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
    or Technical University of Darmstadt, nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
 OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#include "ExperimentConfig.h"
#include "action/ActionModelIK.h"
#include "initState/ISSMiniGolf.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDBodyPosition.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
//#include <TaskDistance1D.h>
#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace Rcs
{

class ECMiniGolf : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Common for the action models
        RcsBody* ground = RcsGraph_getBodyByName(graph, "Ground");
        RCHECK(ground);
        RcsBody* clubTip = RcsGraph_getBodyByName(graph, "ClubTip");
        RCHECK(clubTip);
        RcsBody* ball = RcsGraph_getBodyByName(graph, "Ball");
        RCHECK(ball);
        
        // Get reference frames for the position and orientation tasks
        std::string refFrameType = "world";
        properties->getProperty(refFrameType, "refFrame");
        RcsBody* refBody = nullptr;
        RcsBody* refFrame = nullptr;
        if (refFrameType == "world") {
            // Keep nullptr
        }
        else if (refFrameType == "ball") {
            refBody = ball;
            refFrame = ball;
        }
        else {
            std::ostringstream os;
            os << "Unsupported reference frame type: " << refFrame;
            throw std::invalid_argument(os.str());
        }
        
        if (actionModelType == "ik") {
            // Create the action model
            auto amIK = new AMIKGeneric(graph);
            if (properties->getPropertyBool("positionTasks", true)) {
                amIK->addTask(new TaskPosition1D("X", graph, clubTip, refBody, refFrame));
            }
            else {
                amIK->addTask(new TaskVelocity1D("Xd", graph, clubTip, refBody, refFrame));
            }
            
            // Add fixed tasks after the ones controlled by th policy
            MatNd* fixedClubTipY = MatNd_create(1, 1);
            MatNd* fixedClubTipZ = MatNd_create(1, 1);
            MatNd_set(fixedClubTipY, 0, 0, -0.0);
            MatNd_set(fixedClubTipZ, 0, 0, -0.03);
            amIK->addFixedTask(new TaskPosition1D("Y", graph, ball, clubTip, ground), fixedClubTipY);
            amIK->addFixedTask(new TaskPosition1D("Z", graph, ball, clubTip, ground), fixedClubTipZ);
//            amIK->addFixedTask(new TaskDistance1D(graph, clubTip, ground, 2), fixedClubTipZ);
            return amIK;
        }
        
        else {
            std::ostringstream os;
            os << "Unsupported action model type: " << actionModelType;
            throw std::invalid_argument(os.str());
        }
    }
    
    virtual ObservationModel* createObservationModel()
    {
        auto fullState = new OMCombined();
        
        // Observe the ball's position
        auto omLinBall = new OMBodyStateLinear(graph, "Ball", nullptr); // former: "Ground"
        omLinBall->setMaxVelocity(10.); // [m/s]
        fullState->addPart(omLinBall);
        
        // Observe the club's position
        auto omLinClub = new OMBodyStateLinear(graph, "ClubTip", nullptr); // former: "Ground"
        omLinClub->setMaxVelocity(5.); // [m/s]
        fullState->addPart(omLinClub);
        
        // Observe the club's orientation
        auto omAng = new OMBodyStateAngular(graph, "ClubTip", nullptr); // former: "Ground"
        omAng->setMaxVelocity(20.); // [rad/s]
        fullState->addPart(omAng);
        
        // Observe the robot's joints
        std::list<std::string> listOfJointNames = {"base-m3", "m3-m4", "m4-m5", "m5-m6", "m6-m7", "m7-m8", "m8-m9"};
        for (std::string jointName : listOfJointNames) {
            fullState->addPart(new OMJointState(graph, jointName.c_str(), false));
        }
        
        std::string actionModelType = "unspecified";
        properties->getProperty(actionModelType, "actionModelType");
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true)) {
            RcsSensor* fts = RcsGraph_getSensorByName(graph, "WristLoadCellLBR");
            if (fts) {
                auto omForceTorque = new OMForceTorque(graph, fts->name, 300);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", false) & (collisionMdl != nullptr)) {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add predicted collision cost
        if (properties->getPropertyBool("observePredictedCollisionCost", false) && collisionMdl != nullptr) {
            // Get horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add collision model
            auto omCollisionCost = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, 50);
            fullState->addPart(omCollisionCost);
        }
        
        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", false)) {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK) {
                auto omTSDescr = new OMTaskSpaceDiscrepancy("ClubTip", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescr);
            }
            else {
                delete fullState;
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Ball", new PPDMassProperties());
        manager->addParam("Ball", new PPDMaterialProperties());
        manager->addParam("Club", new PPDMassProperties());
        manager->addParam("Ground", new PPDMaterialProperties());
        manager->addParam("Ground", new PPDMaterialProperties());
        manager->addParam("ObstacleLeft", new PPDBodyPosition(true, true, false));
        manager->addParam("ObstacleRight", new PPDBodyPosition(true, true, false));
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSMiniGolf(graph, properties->getPropertyBool("fixedInitState", true));
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set the camera center
        double cameraCenter[3];
        cameraCenter[0] = 2.0;
        cameraCenter[1] = 1.0;
        cameraCenter[2] = 0.0;
        
        // Set the camera position
        double cameraLocation[3];
        cameraLocation[0] = -4.5;
        cameraLocation[1] = 4.5;
        cameraLocation[2] = 3.5;
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        // Apply camera position
        viewer->setCameraHomePosition(osg::Vec3d(cameraLocation[0], cameraLocation[1], cameraLocation[2]),
                                      osg::Vec3d(cameraCenter[0], cameraCenter[1], cameraCenter[2]),
                                      osg::Vec3d(cameraUp[0], cameraUp[1], cameraUp[2]));
#endif
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* effector = RcsGraph_getBodyByName(graph, "ClubTip");
        RCHECK(effector);
        return new ForceDisturber(effector, effector);
    }
    
    void
    getHUDText(
        std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
        PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }
        
        linesOut.emplace_back(
            string_format("physics engine: %s                            sim time: %2.3f s", simName, currentTime));
        
        unsigned int numPosCtrlJoints = 0;
        unsigned int numTrqCtrlJoints = 0;
        // Iterate over unconstrained joints
        RCSGRAPH_TRAVERSE_JOINTS(graph) {
                if (JNT->jacobiIndex != -1) {
                    if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_POSITION) {
                        numPosCtrlJoints++;
                    }
                    else if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_TORQUE) {
                        numTrqCtrlJoints++;
                    }
                }
            }
        linesOut.emplace_back(
            string_format("num joints:    %d total, %d pos ctrl, %d trq ctrl", graph->nJ, numPosCtrlJoints,
                          numTrqCtrlJoints));

//        unsigned int sd = observationModel->getStateDim();
        
        auto omLinBall = observationModel->findOffsets<OMBodyStateLinear>(); // finds the first OMBodyStateLinear
        if (omLinBall) {
            linesOut.emplace_back(string_format("ball pos:     [% 1.3f,% 1.3f,% 1.3f] m",
                                                obs->ele[omLinBall.pos],
                                                obs->ele[omLinBall.pos + 1],
                                                obs->ele[omLinBall.pos + 2]));
            linesOut.emplace_back(string_format("club tip pos: [% 1.3f,% 1.3f,% 1.3f] m",
                                                obs->ele[omLinBall.pos + 3],
                                                obs->ele[omLinBall.pos + 4],
                                                obs->ele[omLinBall.pos + 5]));
        }
        
        auto omAng = observationModel->findOffsets<OMBodyStateAngular>();
        if (omAng) {
            linesOut.emplace_back(string_format("club tip ang: [% 1.3f,% 1.3f,% 1.3f] deg",
                                                RCS_RAD2DEG(obs->ele[omAng.pos]),
                                                RCS_RAD2DEG(obs->ele[omAng.pos + 1]),
                                                RCS_RAD2DEG(obs->ele[omAng.pos + 2])));
        }
        
        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT) {
            linesOut.emplace_back(
                string_format("forces:       [% 3.1f,% 3.1f,% 3.1f] N",
                              obs->ele[omFT.pos], obs->ele[omFT.pos + 1], obs->ele[omFT.pos + 2]));
        }
        
        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD) {
            linesOut.emplace_back(
                string_format("ts delta:     [% 1.3f,% 1.3f] m", obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1]));
        }
        
        std::stringstream ss;
        ss << "actions:      [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++) {
            ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6) {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
        
        if (physicsManager != nullptr) {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* ball_bpi = physicsManager->getBodyInfo("Ball");
            BodyParamInfo* club_bpi = physicsManager->getBodyInfo("Club");
            BodyParamInfo* ground_bpi = physicsManager->getBodyInfo("Ground");
            
            linesOut.emplace_back(
                string_format("ball mass:     %1.2f kg                            club mass: %1.2f kg",
                              ball_bpi->body->m, club_bpi->body->m));
            linesOut.emplace_back(string_format("ball friction: %1.3f                        ground friction: %1.3f",
                                                ball_bpi->material.getFrictionCoefficient(),
                                                ground_bpi->material.getFrictionCoefficient()));
            linesOut.emplace_back(string_format("ball rolling friction: %1.3f        ground rolling friction: %1.3f",
                                                ball_bpi->material.getRollingFrictionCoefficient(),
                                                ground_bpi->material.getRollingFrictionCoefficient()));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECMiniGolf> RegMiniGolf("MiniGolf");
    
}
