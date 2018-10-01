/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 200;
	
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	
	normal_distribution<double> dist_x(x, std_x);    //两个参数，均值和方差
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	default_random_engine gen;
	
	for (int i = 0; i < num_particles; i++)
	{
		Particle p;    //Particle是一个粒子类，p是一个粒子对象

		p.id = i;    //粒子的编号
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);	 
		 
		p.weight = 1.0;    //粒子的权重，初始化为1.0
		 
		particles.push_back(p);    //particles是粒子的集合，依次向其中添加粒子
		 
		 
		weights.push_back(1.0);    //weights是所有粒子的权重构成的向量，依次向其中添加粒子权重
	}
	is_initialized = true;

}


// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	
	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	default_random_engine gen;

	for (int i = 0; i < num_particles; i++)
	{
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		if (fabs(yaw_rate) <= 0.001)
		{
			x = x + velocity*delta_t*(cos(theta));
			y = y + velocity*delta_t*(sin(theta));
		}else
		{
			x = x + velocity/yaw_rate*( sin(theta + yaw_rate*delta_t) - sin(theta));    //yaw_rate非0的时候，采用公式计算
			y = y + velocity/yaw_rate*( cos(theta) - cos( theta + yaw_rate*delta_t));
			theta = theta + yaw_rate*delta_t;
		}
		
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}


// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	for (int i=0; i< observations.size();i++)
	{
		LandmarkObs obs = observations[i];    //LandmarkObs是一个在help_functions.h中定义的struct类型
		int landmark_id = 0;
		double minimum_dist;
		
		for (int j=0;j<predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			
			//dist函数计算两点之间的欧几里得距离，在help_functions.h中定义，Computes the Euclidean distance between two 2D points.
			double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
			
			if (j==0)
			{
				minimum_dist = current_dist;
			}
			
			if (current_dist <= minimum_dist)
			{
				minimum_dist = current_dist;
				landmark_id = pred.id;
			}
		}
		observations[i].id = landmark_id;
	}
}

// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	for (int particle_num = 0; particle_num < num_particles; particle_num ++)
	{
		double ptcl_x = particles[particle_num].x;
		double ptcl_y = particles[particle_num].y;
		double ptcl_theta = particles[particle_num].theta;
		
		//将测量值转换到地图坐标系中transform the observations to the map co-ordinate system
		vector<LandmarkObs> trans_obs;    //LandmarkObs是一个在help_functions.h中定义的struct类型
		
		double trans_x,trans_y;
		
		for(int i = 0; i < observations.size(); i++)
		{
			//变换x坐标
			trans_x = ptcl_x + (cos(ptcl_theta) * observations[i].x) - (sin(ptcl_theta) * observations[i].y);
			//变换y坐标
			trans_y = ptcl_y + (sin(ptcl_theta) * observations[i].x) + (cos(ptcl_theta) * observations[i].y);
			//追加到vector中
			trans_obs.push_back(LandmarkObs{observations[i].id, trans_x, trans_y});
		}

		vector<LandmarkObs> predicted_landmarks;
		double diff_x, diff_y, lndmrk_x, lndmrk_y;
		
		for(int landmark_iterator = 0; landmark_iterator < map_landmarks.landmark_list.size(); landmark_iterator++)
		{
			lndmrk_x = map_landmarks.landmark_list[landmark_iterator].x_f;
			lndmrk_y = map_landmarks.landmark_list[landmark_iterator].y_f;
			
			diff_x = fabs(lndmrk_x - ptcl_x);
			diff_y = fabs(lndmrk_y - ptcl_y);
			if ((diff_x <= sensor_range) && (diff_y <= sensor_range))
			{
				predicted_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[landmark_iterator].id_i, lndmrk_x, lndmrk_y});
			}
		}

		dataAssociation(predicted_landmarks, trans_obs);
	
		//更新权重
		particles[particle_num].weight = 1.0;
		for (int j = 0; j < trans_obs.size(); j++)
		{
			double trans_obs_x, trans_obs_y, pred_x, pred_y;
			trans_obs_x = trans_obs[j].x;
			trans_obs_y = trans_obs[j].y;
			int associated_prediction = trans_obs[j].id;

			for (int k = 0; k < predicted_landmarks.size(); k++)
			{
				if (predicted_landmarks[k].id == associated_prediction)
				{
					pred_x = predicted_landmarks[k].x;
					pred_y = predicted_landmarks[k].y;
				}
			}
			
			//代入公式计算
			double obs_wgt_each_lndmrk = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) 
											* exp(-(pow(pred_x-trans_obs_x, 2)/(2*pow(std_landmark[0], 2)) 
											+ (pow(pred_y-trans_obs_y, 2)/(2*pow(std_landmark[1], 2)))));
			
			particles[particle_num].weight *= obs_wgt_each_lndmrk;
			
			weights[particle_num] = particles[particle_num].weight;
		}
	}
}


// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */

void ParticleFilter::resample()
{
	default_random_engine generator;
	uniform_real_distribution<double> uniform(0, 1);
	
	vector<Particle> resampled;
	double max_weight = 0.0;
	for(int i=0; i<num_particles; i++)
	{
		if(weights[i] > max_weight)
		{
			max_weight = weights[i];
		}
	}
	
	int index = (int)(uniform(generator) * num_particles);
	double beta = 0.0;
	for(int i=0; i<num_particles; i++)
	{
		beta += uniform(generator) * 2 * max_weight;
		
		while(beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled.push_back(particles[index]);
	}
	
	particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
