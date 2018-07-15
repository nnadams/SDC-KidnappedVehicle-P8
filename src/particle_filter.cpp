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

static default_random_engine rengine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  normal_distribution<double> norm_x(x, std[0]);
  normal_distribution<double> norm_y(y, std[1]);
  normal_distribution<double> norm_theta(theta, std[2]);

  num_particles = 150;
  for (int i = 0; i < num_particles; i++) {
    Particle ptcl;

    ptcl.id = i;
    ptcl.x = norm_x(rengine);
    ptcl.y = norm_y(rengine);
    ptcl.theta = norm_theta(rengine);
    ptcl.weight = 1.0;

    particles.push_back(ptcl);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> norm_x(0, std_pos[0]);
  normal_distribution<double> norm_y(0, std_pos[1]);
  normal_distribution<double> norm_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle* ptcl = &particles[i];

    // Update the position and the angle
    if (fabs(yaw_rate) > 0.00001) { // don't forget the absolute value!
      ptcl->x += ((velocity / yaw_rate) * (sin(ptcl->theta + (yaw_rate * delta_t)) - sin(ptcl->theta)));
      ptcl->y += ((velocity / yaw_rate) * (cos(ptcl->theta) - cos(ptcl->theta + (yaw_rate * delta_t))));
      ptcl->theta += (yaw_rate * delta_t);
    }
    else {
      ptcl->x += velocity * delta_t * cos(ptcl->theta);
      ptcl->y += velocity * delta_t * sin(ptcl->theta);
    }

    // Add random noise to emulate "sensor noise"
    ptcl->x += norm_x(rengine);
    ptcl->y += norm_y(rengine);
    ptcl->theta += norm_theta(rengine);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (uint i = 0; i < observations.size(); i++) {
    LandmarkObs *obs = &observations[i];
    int min_distance = numeric_limits<int>::max();
    int minId = -1;

    for (uint j = 0; j < predicted.size(); j++) {
      LandmarkObs *pred = &predicted[j];

      int distance = dist(obs->x, obs->y, pred->x, pred->y);
      if (distance < min_distance) {
        min_distance = distance;
        minId = pred->id;
      }
    }

    obs->id = minId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  double s_x = std_landmark[0];
  double s_y = std_landmark[1];

  for (int i = 0; i < num_particles; i++) {
    Particle *ptcl = &particles[i];
    vector<LandmarkObs> predictions;
    vector<LandmarkObs> obs_transformed;

    // List all landmarks within range
    for (uint j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double lmk_x = map_landmarks.landmark_list[j].x_f;
      double lmk_y = map_landmarks.landmark_list[j].y_f;
      int lmk_id = map_landmarks.landmark_list[j].id_i;

      if (dist(ptcl->x, ptcl->y, lmk_x, lmk_y) <= sensor_range) {
        predictions.push_back(LandmarkObs {lmk_id, lmk_x, lmk_y});
      }
    }

    // Transform sensor observations from vehicle coordinates to the map coordinates
    for (uint j = 0; j < observations.size(); j++) {
      double transformed_x = ptcl->x + (cos(ptcl->theta) * observations[j].x) - (sin(ptcl->theta) * observations[j].y);
      double transformed_y = ptcl->y + (sin(ptcl->theta) * observations[j].x) + (cos(ptcl->theta) * observations[j].y);
      obs_transformed.push_back(LandmarkObs {observations[j].id, transformed_x, transformed_y});
    }

    // Use nearest neighbor to match each observation to a nearby landmark
    dataAssociation(predictions, obs_transformed);

    ptcl->weight = 1.0;

    // Update this particle's weight using each observation and matching landmark
    for (uint j = 0; j < obs_transformed.size(); j++) {
      double o_x = obs_transformed[j].x;
      double o_y = obs_transformed[j].y;
      double p_x, p_y;
      for (uint k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == obs_transformed[j].id) {
          p_x = predictions[k].x;
          p_y = predictions[k].y;
        }
      }

      // Using multivariate gaussian formula
      double gauss_norm = (1 / (2 * M_PI * s_x * s_y));
      double exponent = pow(p_x - o_x, 2) / (2*pow(s_x, 2)) + (pow(p_y-o_y,2) / (2*pow(s_y, 2)));

      ptcl->weight *= gauss_norm * exp(-exponent);
    }

    weights[i] = ptcl->weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> new_particles;
  vector<double> new_weights;

  for (int i = 0; i < num_particles; i++) {
    Particle *ptcl = &particles[d(rengine)];

    new_particles.push_back(*ptcl);
    new_weights.push_back(ptcl->weight);
  }

  particles = new_particles;
  weights = new_weights;
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

    return particle;
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

