# Remove all variables from the R environment to create a fresh start
rm(list=ls())
# Execute our custom script for loading packages
source("usePackages.R")
# Name of the packages 
pkgnames <- c("simmer", "simmer.plot", "GA")
# Use our custom load function
loadPkgs(pkgnames)

# Seed for reproducibility
set.seed(42)


# Define function to simulate outpatient clinic and calculate objective scores
sim_clinic <- function(num_resources) {
  num_receptionists <- num_resources[1]
  num_nurses  <- num_resources[2]
  num_doctors <- num_resources[3]
  num_pharmacists <- num_resources[4]
  
  # Create environment
  env2 <- simmer("outpatient clinic")
  
  # Add resources with manual capacity adjustment
  env2 <- env2 %>%
    add_resource("receptionist", capacity = num_receptionists) %>% 
    add_resource("nurse", capacity = num_nurses) %>% 
    add_resource("doctor", capacity = num_doctors) %>% 
    add_resource("pharmacy", capacity = num_pharmacists) 
  
  # Define trajectory with cost tracking
  patient <- trajectory(name = "Patient Path", verbose = TRUE) %>%  
    seize("receptionist", 1) %>% 
    timeout(function() runif(1, min = 5, max = 7)) %>%  
    release("receptionist", 1) %>%  
    seize("nurse", 1) %>%
    timeout(function() rnorm(1, 10, 2)) %>%
    release("nurse", 1) %>%
    seize("doctor", 1) %>%
    timeout(function() rnorm(1, 15, 3)) %>%
    release("doctor", 1) %>%
    seize("pharmacy", 1) %>%
    timeout(function() rnorm(1, 8, 1)) %>%
    release("pharmacy", 1) %>% 
    seize("receptionist", 1) %>%
    timeout(function() runif(1, min = 5, max = 7)) %>%
    release("receptionist", 1) %>%
    set_attribute("start_time", function() now(env2))  
  
  # Create patient generator
  env2 %>%
    add_generator(
      name_prefix = "patient_",
      trajectory = patient,
      distribution = function()  rnorm(1,5,1) 
    )
  
  # Run simulation
  total_minutes <- 480
  env2 %>% run(until = total_minutes)
  
  # Get monitor arrivals
  arrivals <- get_mon_arrivals(env2)
  
  # Get monitor resources
  resources <- get_mon_resources(env2)
  
  # Define costs associated with resources
  resource_costs <- list(receptionist = 110, nurse = 160, doctor = 250,  pharmacy=130)
  
  # Calculate total manpower cost
  total_manpower_cost <- resource_costs$receptionist * num_receptionists +
    resource_costs$nurse * num_nurses +
    resource_costs$doctor * num_doctors +
    resource_costs$pharmacy * num_pharmacists
  
  
  # Calculate total cost associated with waiting time
  waiting_cost <- sum(( arrivals$end_time - arrivals$start_time) -  arrivals$activity_time ) * ((resource_costs$nurse/total_minutes)*0.5) 
  
  # Calculate the total number of patients served
  patients_served <- length(arrivals$name)
  
  # Return waiting cost, total manpower cost, and total patients served
  return(c(waiting_cost, total_manpower_cost, patients_served))
}


# Define function to evaluate objective scores for each solution in population
fitness_function <- function(num_resources) {
  objectives <- sim_clinic(num_resources)
  
  # Define weights for each objective
  waiting_cost_weight <- 0.25  # Adjust weight based on relative importance
  manpower_cost_weight <- 0.25  # Adjust weight based on relative importance
  num_patients_weight <- 0.5   # Adjust weight based on relative importance
  
  # Compute aggregate fitness score as a weighted sum of objectives
  fitness_score <- (- (waiting_cost_weight * objectives[1] ) - 
                      manpower_cost_weight * objectives[2] +  
                      num_patients_weight * objectives[3])
  
  return(fitness_score)
}

# Define the genetic algorithm parameters
num_resources_range <- matrix(c(1, 1, 1, 1, 5, 5, 5, 5), ncol = 2)  # Range of possible values for each resource
popSize <- 50  # Population size
maxiter <- 15  # Maximum number of generations

# Run genetic algorithm
ga_result <- ga(type = "real-valued", 
                fitness = fitness_function, 
                lower = num_resources_range[, 1], 
                upper = num_resources_range[, 2],
                popSize = popSize,
                maxiter = maxiter,
                pcrossover = 0.8,
                pmutation = 0.1)

# Print best solution
best_solution <- round(ga_result@solution[1, ])
cat("Best solution (Number of receptionists, nurses, doctors and  pharmacists):", best_solution, "\n")

# Evaluate objectives for the best solution
best_objectives <- sim_clinic(best_solution)
cat("Total manpower cost for best solution:", best_objectives[2], "\n")
cat("Waiting cost for best solution:", best_objectives[1], "\n")
cat("Total number of patients served for best solution:", best_objectives[3], "\n")