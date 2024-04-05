classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end




function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1   

            % % Implementation of Nearest Neighbour Filter , full recursion         
            totalTrackTime = size(Z,1);
            log_factor = log(sensormodel.P_D / sensormodel.lambda_c ) ;
            log_wk_zero = log ( 1 - sensormodel.P_D ) ;

            % Possible outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x_P = cell(totalTrackTime, 1);

            % useful parameters
            log_detect_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_missed  = log(1 - sensormodel.P_D);

            % iterate through timestamps
            for i = 1 : totalTrackTime

                % get current timestep measurements
                z_i = Z{i};

                % perform gating and find number of measurements inside limits
                [z_ingate, ~] = obj.density.ellipsoidalGating(state, z_i, measmodel, obj.gating.size);
                
                if ~isempty(z_ingate)
                    likelihoodDensity = obj.density.predictedLikelihood(state, z_ingate, measmodel); 
                    log_weights = log_detect_factor + likelihoodDensity;

                    [max_weight, index] = max(log_weights);
                    if(max_weight > log_missed) 
                        % Kalman update
                        state = obj.density.update(state, z_ingate(:, index) , measmodel);
                    end
                end

                % updated state variables 
                estimates{i}   = state.x;
                estimates_x{i} = state.x;

                % predict the next state
                state = obj.density.predict(state, motionmodel);
            end	

end



     
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)

            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  



             % % Implementation of Probably Ass Filter , full recursion         
            totalTrackTime = size(Z,1);
            log_factor = log(sensormodel.P_D / sensormodel.lambda_c ) ;
            log_wk_zero = log ( 1 - sensormodel.P_D ) ;

            % Possible outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x_P = cell(totalTrackTime, 1);

            % useful parameters
            log_detect_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_missed  = log(1 - sensormodel.P_D);

     
            %%%%%
             
              for i = 1 : totalTrackTime
                z = Z{i} ;
                % gating;
                [ z_ingate , ~ ] = obj.density.ellipsoidalGating( state , z , measmodel , obj.gating.size );
                % create missed detection hypothesis;
                mk = size(z_ingate , 2 ) + 1 ;
                multiHypotheses_new = repmat(state, mk , 1) ;
                % create object detection hypotheses for each detection inside the gate;
                log_weights = log_detect_factor + obj.density.predictedLikelihood(state, z_ingate, measmodel);
                log_weights = [ log_weights ; log_missed] ;
                for k = 1 : mk - 1 
                    multiHypotheses_new(k,1) = obj.density.update( multiHypotheses_new(k,1) , z_ingate(:,k) , measmodel );
                end
                % normalise hypothesis weights;
                log_weights = normalizeLogWeights(log_weights) ;
                % prune hypotheses with small weights, and then re-normalise the weights.
                [log_weights , multiHypotheses_new ] = ...
                    hypothesisReduction.prune(log_weights , multiHypotheses_new , obj.reduction.w_min ) ;
                % merge different hypotheses using Gaussian moment matching;
                [log_weights , multiHypotheses_new ] = ...
                    hypothesisReduction.merge(log_weights , multiHypotheses_new , realmax , obj.density ) ;
                % extract object state estimate. At this point multiHypotheses is be a single array
                state = multiHypotheses_new(1,1) ;
                estimates{i} = state.x;
                estimates_x_P{i} = state;
                % prediction.
                state = obj.density.predict(state , motionmodel ) ;
                
              end

              %%%%%%%%%%%%%%%% END OF FUNCTION %%%%%%%%%%%%%%%%%%%
          
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
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

            %       state dimension) x 1  


            % % Implementation of Gaussian Sum Filter , full recursion         
            totalTrackTime = size(Z,1);

            % Possible outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x_P = cell(totalTrackTime, 1);

            % useful parameters
            LOG_DETECT = log(sensormodel.P_D / sensormodel.intensity_c);
            LOG_MISSED  = log(1 - sensormodel.P_D);

            % Hypotheses arrays. Suppose we start with a single one
            multiHypotheses = repmat(state , 1 , 1 ) ;
            numHypotheses = size(multiHypotheses , 1) ;
            log_weights = repmat( 0 , 1 , 1 )   ;


            for i = 1 : totalTrackTime
                z_i = Z{i};
                % Allocate space for new hypotheses
                multiHypotheses_new = [] ;
                log_weights_new = [] ;

                % For each hypotheses perform gating and update
                for k = 1 : numHypotheses
                 [ z_ingate , ~ ] = ...
                 obj.density.ellipsoidalGating( multiHypotheses(k,1) , z_i , measmodel , obj.gating.size );
                   num_ingate = size(z_ingate , 2 );
                   if num_ingate > 0
                      for j = 1 : num_ingate
                           log_weights_new = [ log_weights_new  ; LOG_DETECT + log_weights(k,1) +  obj.density.predictedLikelihood( multiHypotheses(k,1) , z_ingate(:,j) , measmodel) ] ;
                           multiHypotheses_new = [ multiHypotheses_new  ; obj.density.update( multiHypotheses(k,1) , z_ingate(:,j), measmodel )  ] ;
                      end
                   end
                end
                % normalise hypothsis weights;
                log_weights = [ log_weights_new ; log_weights + LOG_MISSED ] ;
                multiHypotheses = [ multiHypotheses_new ; multiHypotheses ] ;
                log_weights = normalizeLogWeights(log_weights) ;

                % prune hypotheses with small weights, and then re-normalise the weights;
                [log_weights , multiHypotheses ] = ...
                    hypothesisReduction.prune(log_weights , multiHypotheses , obj.reduction.w_min );
                  log_weights = normalizeLogWeights(log_weights ) ;



                % hypothesis merging (to achieve this, you just need to directly call function hypothesisReduction.merge.);
                [log_weights ,multiHypotheses ] =  ...
                 hypothesisReduction.merge( log_weights  , multiHypotheses , obj.reduction.merging_threshold ,  obj.density);
                 log_weights = normalizeLogWeights(log_weights ) ;

                % cap the number of the hypotheses, and then re-normalise the weights;
                [log_weights , multiHypotheses ] = ...
                 hypothesisReduction.cap(log_weights , multiHypotheses , obj.reduction.M);
                 log_weights = normalizeLogWeights(log_weights ) ;

                  % extract object state estimate using the most probable hypothesis estimation;
                  [~, indexes] = max(log_weights);                       
                  % Possible outputs
                  extracted_struct =  multiHypotheses(indexes(1),1) ;
                  estimates{i} = extracted_struct.x ;
                  estimates_x_P{i} = extracted_struct ;
                  % for each hypothesis, perform prediction.
                   numHypotheses = size(multiHypotheses,1) ;
                  for j = 1 : numHypotheses
                     multiHypotheses(j,1) = obj.density.predict( multiHypotheses(j,1) , motionmodel) ;
                  end
            end

        
                    %%%%%%%%%% end of gsf


    



        end
                    %%%%%%%%%% end of gsf
















% 

%%%%%%%%%%%%%%



    end
end

