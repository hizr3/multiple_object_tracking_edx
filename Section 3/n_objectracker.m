classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            %total time
            totalTrackTime = numel(Z);
            % results
            estimates = cell(totalTrackTime,1);
            num_objects = size(states , 2 );


            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);

            for k = 1 : 1
                % Perform gating
                z_k = Z{k};
                num_meas_k = size(z_k , 2 );
                assignments = zeros(num_objects , num_meas_k);
                for i = 1 : num_objects
                    [~, index] = obj.density.ellipsoidalGating( states(i) , z_k, measmodel, obj.gating.size);
                    assignments(i,:) = index';
                end
                assignments_ingate = sum( assignments , 1 ) > 0 
                % Compute measurements in gate
                z_k_ingate  = z_k(:,assignments_ingate) ;
                num_meas_k = size(z_k_ingate , 2 );

                costMatrix = inf(num_objects , num_meas_k + num_objects)
                 for i = 1 : num_objects
                     for j = 1 : num_meas_k
                         lij = 1;
                         L(i,j) = lij;
                     end
                 end
                 L(:,num_meas_k : num_meas_k + num_objects ) = 

            end
            % end for


        end
    end
end

