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
      
            % Implementation of Nearest Neighbour Filter, full recursion %

            %%%%%%%%%%%%%%%%%%%%%%%%
              N = numel ( Z ) ; 

              dim_state = size ( state.x , 1 ) ; 
              dim_meas = size(Z{1},1) ;
              

              state_hat = cell (  N , 1 );
              state_hat{1} = state;
              estimates = cell( N , 1) ;
              estimates{1} = state.x ;
              
              % useful parameters
              log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
              wk_theta_zero  = 1 - sensormodel.P_D;
              
              for i = 1 : N

                  z = Z{i} ; 
                  % Gating
                  [ z_ingate , meas_in_gate ] = obj.density.ellipsoidalGating( state , z , measmodel , obj.gating.size );
                  
                  % Number of hypotheses ( num meas + false detect hypothesis ) 
                  mk = 1 + size(z_ingate,2);
                   
                  % If mk = 1 the state updated is skipped
                  if mk > 1 
                    likelihoodDensity = obj.density.predictedLikelihood(state, z_ingate, measmodel); 
                    % The update formula is expressed through the log
                    % likelihood for ease of use. remember that wk_factor
                    % is P_D/intensity
                    wk_theta = exp(log_wk_theta_factor + likelihoodDensity);                    
                    % Find max hypothesis
                    [max_wk_theta, index] = max(wk_theta);
                    if(max_wk_theta >= wk_theta_zero) % If clutter intensity is greater, we skip the update step              
                        state = obj.density.update(state, z_ingate(:,index), measmodel);
                    end

                  end                            
                  % Possible outputs
                  state_hat{i} = state;
                  estimates{i} = state.x ;

                  % Predict
                  state = obj.density.predict(state, motionmodel);  
              end

              %%%%%%%%%%%%%%%% END OF FUNCTION %%%%%%%%%%%%%%%%%%%
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
       
            %%%%%%%%%%%%%%% Implementation of Probabilistic Ass Filter,
            %%%%%%%%%%%%%%% full recursion
             N = numel ( Z ) ; 
              estimates = cell( N , 1) ;

              % useful parameters
              log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
              log_wk_theta_zero  = log(1 - sensormodel.P_D); % Store it in log form for normalization purposes

              for i = 1 : N

                  z = Z{i} ; 
                  % Gating
                  [ z_ingate , meas_in_gate ] = obj.density.ellipsoidalGating( state , z , measmodel , obj.gating.size );
                  mk = size(z_ingate,2);
                    % need to use repmat instead of cell for
                                % some compatibility issues 
                   hypo_0_mk = repmat(state,mk + 1,1);                        
                                log_wk_theta_1_mk = log_wk_theta_factor + ...
                                obj.density.predictedLikelihood(state, z_ingate, measmodel);
                                log_weights = normalizeLogWeights([log_wk_theta_1_mk ; log_wk_theta_zero]) ;                   
                                for k = 1 : mk
                                  hypo_0_mk(k,1) = obj.density.update( state, z_ingate(:,k), measmodel );
                                end
                                hypo_0_mk(mk + 1 , 1) = state;
                                
                                 % Mixture reduction
                                 % Cap

                                 [log_weights, hypo_0_mk] = ...
                            hypothesisReduction.cap(log_weights, hypo_0_mk, obj.reduction.M);
                            log_weights = normalizeLogWeights(log_weights) ;                   


                                 % Prune
                                 %disp(log_weights)
                                 %disp(hypo_0_mk)
                                 [log_weights, hypo_0_mk] = ...
                            hypothesisReduction.prune(log_weights, hypo_0_mk, obj.reduction.w_min);
                                log_weights = normalizeLogWeights(log_weights) ;                   
                      
                             [log_weights,hypo_0_mk] =  hypothesisReduction.merge(...
                       log_weights,...
                        hypo_0_mk,...
                     100000000000000000000000000 ,...
                        obj.density);
                    

                  % When using repmat the second index is
                  % the ficticious dimension
                  state = hypo_0_mk(1,1);
                  estimates{i} = state.x ;

                  % Predict
                  state = obj.density.predict(state, motionmodel);  
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
                N = numel ( Z ) ; 
               estimates = cell( N , 1) ;
              % useful parameters
              log_wk_theta_factor = log( sensormodel.P_D / sensormodel.intensity_c );
              log_w0_factor  = log( 1 - sensormodel.P_D ) ; % Store it in log form for normalization purposes

              % Initiliaze all the hypotheses to the maximum number
                multiHypotheses_old = repmat(state,1,1) ;
                logWeights_old = repmat(0,1,1) ;
                numHypotheses = 1 ;

              for i = 1 : N
                %for each hypothesis, create missed detection hypothesis
                %for each hypothesis, perform ellipsoidal gating and only create object detection hypotheses for detections inside the gate;
                multiHypotheses_new = multiHypotheses_old;
                logWeights_new = logWeights_old + repmat(log_w0_factor,size(logWeights_old,1) , 1); % its fine to sum structs here. Add the
                % current measurement
                 z = Z{i};
                 %disp(z);
                for k = 1 : numHypotheses
                    angelinaMango = multiHypotheses_old(k,1);
                    [ z_ingate , ~ ] = obj.density.ellipsoidalGating( angelinaMango  , z , measmodel , obj.gating.size ); 
                 mk = size(z_ingate,2);
                 if mk > 0
                    for j = 1 : mk
                        multiHypotheses_new = [ multiHypotheses_new ; obj.density.update( multiHypotheses_old(k,1) , z_ingate(:,j), measmodel ) ] ;
                        logWeights_new = [ logWeights_new  ; logWeights_old(k,1) + obj.density.predictedLikelihood( multiHypotheses_old(k,1) , z_ingate(:,j) , measmodel) ] ;
                    end
                 end
                 %normalise hypothsis weights;
                 logWeights_new = normalizeLogWeights(logWeights_new) ;
                 % prune hypotheses with small weights, and then re-normalise the weights;
                 [logWeights_new , multiHypotheses_new ] = hypothesisReduction.prune(logWeights_new , multiHypotheses_new , obj.reduction.w_min);
                 logWeights_new = normalizeLogWeights(logWeights_new) ;                   
                % hypothesis merging (to achieve this, you just need to directly call function hypothesisReduction.merge.);
                 [logWeights_new,multiHypotheses_new] =  hypothesisReduction.merge( logWeights_new ,   multiHypotheses_new, obj.reduction.merging_threshold ,  obj.density);
                  % cap the number of the hypotheses, and then re-normalise the weights;
                 [logWeights_new, multiHypotheses_new] = hypothesisReduction.cap(logWeights_new, multiHypotheses_new, obj.reduction.M);
                  logWeights_new = normalizeLogWeights(logWeights_new) ;
                 % extract object state estimate using the most probably hypothesis estimation;
                  [maxWeight, index] = max(logWeights_new);                       
                  % Possible outputs
                  newStateStruct =  multiHypotheses_new(index,1);
                  estimates{i} = newStateStruct.x ;
                 % for each hypothesis, perform prediction
                    numHypotheses = size(multiHypotheses_new,1) ;
                     for j = 1 : numHypotheses
                         multiHypotheses_new(j,1) = obj.density.predict( multiHypotheses_new(j,1) , motionmodel) ;
                     end
                   
                    logWeights_old = logWeights_new ;
                    multiHypotheses_old = multiHypotheses_new ;
                    clear logWeights_new multiHypotheses_new ;
                end
              end
        end
        
    end
end

