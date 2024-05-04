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
      

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
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

            trackTime = numel(Z);
            estimates  = cell(trackTime , 1);
            num_objects = size(states , 2 );
            log_detect = log(sensormodel.P_D / sensormodel.intensity_c);
            log_miss  = log(1 - sensormodel.P_D);

            %works
            
            % for each local hypothesis in each hypothesis tree: 
            % 1). implement ellipsoidal gating; 
            % 2). calculate missed detection and predicted likelihood for
            %     each measurement inside the gate and make sure to save
            %     these for future use; 
            % 3). create updated local hypotheses and make sure to save how
            %     these connects to the old hypotheses and to the new the 
            %     measurements for future use;
            %
            % for each predicted global hypothesis: 
            % 1). create 2D cost matrix; 
            % 2). obtain M best assignments using a provided M-best 2D 
            %     assignment solver; 
            % 3). update global hypothesis look-up table according to 
            %     the M best assignment matrices obtained and use your new 
            %     local hypotheses indexing;
            %
            % normalise global hypothesis weights and implement hypothesis
            % reduction technique: pruning and capping;
            %
            % prune local hypotheses that are not included in any of the
            % global hypotheses;
            %
            % Re-index global hypothesis look-up table;
            %
            % extract object state estimates from the global hypothesis
            % with the highest weight;
            %
            % predict each local hypothesis in each hypothesis tree.
            
            local_trees = cell(1 , num_objects)
            for k = 1 : num_objects
                local_trees{k} = states(i);
            end

            for k = 1 : 1 % trackTime
                z_k = Z{k}
                mk = size(z_k , 2)

            end
            
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ estimates  ] = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)


            % GNNFILTER tracks n object using global nearest neighbor
            % association 
            % INPUT: obj: an instantiation of n_objectracker class
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
            % OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)


            totalTrackTime = numel(Z);
            estimates = cell(totalTrackTime,1);
            num_objects = size(states , 2 );
            log_detect = log(sensormodel.P_D / sensormodel.intensity_c);
            log_miss  = log(1 - sensormodel.P_D);

            isEq_  = cell ( totalTrackTime , 1 )



%%% d'rtenzio constants
     N = length(states);
            numSteps = length(Z);
            mis_cost = -log(1-sensormodel.P_D);
%%%

            for k = 1 : totalTrackTime
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55555%%%%%%%%%%%%%%%%%%%%%%%%%

                z_k = Z{k};
                num_meas_k = size(z_k , 2 );
                assignments = zeros(num_objects , num_meas_k);
                for i = 1 : num_objects
                    [~, index] = obj.density.ellipsoidalGating( states(i) , z_k, measmodel, obj.gating.size);
                    assignments(i,:) = index';
                end
                assignments_ingate = sum( assignments , 1 ) > 0 ;
                assignments = assignments(: , assignments_ingate );
                z_k_ingate  = z_k(:,assignments_ingate);
                num_meas_k = size(z_k_ingate , 2 );
                L = inf(num_objects , num_meas_k + num_objects);

                 for i = 1 : num_objects
                     for j = 1 : num_meas_k
                         if assignments(i,j) == 1 
                         H_ih = measmodel.H(states(i).x);
                         S_ih = H_ih*states(i).P*H_ih' + measmodel.R;
                         inn_ih = z_k_ingate(:,j) - measmodel.h(states(i).x);
                         lij = -log_detect ;
                         lij = lij + 0.5*log(det(2*pi*S_ih));
                         lij = lij + 0.5*((inn_ih)')*inv(S_ih)*(inn_ih);
                         L(i,j) = lij;
                         end
                     end
                 end

                 L_miss = L(: , num_meas_k +1  : num_meas_k + num_objects  );
                 L_miss(boolean(eye(num_objects))) = -log_miss*ones(1,num_objects);
                 L(:,num_meas_k + 1 : num_meas_k + num_objects) =  L_miss;

                 
                    

                 %%%%%%%% D'ORTENZIO GNN cost matrix

                z = cell2mat(Z(k));
                M = size(z,2);

                %Ellipsoidal Gating
                [z_ingate, meas_in_gate] = arrayfun(@(state) GaussianDensity.ellipsoidalGating(state,z,measmodel,obj.gating.size),states,'UniformOutput',false);
                %Cost Matrix
                L = Inf(N,M+N); %We initialise for convenience a matrix of Inf elements
                for i=1:N
                    z_ingate_i = cell2mat(z_ingate(i));
                    meas_in_gate_i = cell2mat(meas_in_gate(i));
                    if ~isempty(z_ingate_i)
                        L(i,meas_in_gate_i) = -(log(sensormodel.P_D) + GaussianDensity.predictedLikelihood(states(i),z_ingate_i,measmodel) - log(sensormodel.intensity_c));
                    end
                    L(i,M+i) = -mis_cost;
                end
                %Reduction of the cost matrix
                indexes = find(sum(cell2mat(meas_in_gate),2) == 0); %With this command we find the indexes of the measurements not falling in any gate
                L(:,indexes) = [];
                z(:,indexes) = [];
                % 
                % CON QUESTA ISTRUZIONE ALESSANDRO TROVA GLI INDICI UGUALI
                % A ZERO , E IMPOSTA POI GLI ELEMENTI AD UN ARRAY VUOTO
                % 
                 %%%%%%%% L'ERRORE NON SEMBRA ESSERE NELLA MATRICE DI COSTO
                 %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


states_backup  =states ;
                 % Find assigment
                 [col4row,~,gain]=assign2D(L);
                    if gain == -1
                        disp("unfeasable at step " + num2str(k))
                    end


                 for i = 1 : num_objects
                     if col4row(i) <= num_meas_k
                       states_backup(i) = obj.density.update(states_backup(i), z_k_ingate(:,col4row(i) ), measmodel);
                     end                 
                    estimates{k}(:,i) = states(i).x;
                    states_backup(i) = obj.density.predict( states_backup(i) , motionmodel);
                 end

                states_backup ;

                 %%% D'ORTENZIO CODE


                    [col4row,~,~] = assign2D(L);
                %We now create the new local hypotheses
                if ~isempty(z)
                    for i=1:N
                        if col4row(i)<=size(z,2)
                            states(i) = GaussianDensity.update(states(i),z(:,col4row(i)),measmodel);
                        end
                    end
                end
                estimates{k} = [states.x];
                states = arrayfun(@(state) GaussianDensity.predict(state,motionmodel), states);
                
                states ;
                 %%%
              isEq_{k} = isequal(states , states_backup);
            end          
isEq_
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function estimates = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %JPDAFILTER tracks n object using joint probabilistic data
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

            % 1. implement ellipsoidal gating for each local hypothesis
            % seperately; OK
            % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
            % 3. find the M best assignment matrices using a M-best 2D assignment solver;
            % 4. normalise the weights of different data association hypotheses;
            % 5. prune assignment matrices that correspond to data association hypotheses with low weights and renormalise the weights;
            % 6. create new local hypotheses for each of the data association results;
            % 7. merge local hypotheses that correspond to the same object by moment matching;
            % 8. extract object state estimates;
            % 9. predict each local hypothesis.

            totalTrackTime = numel(Z);
            estimates = cell(totalTrackTime,1);
            num_objects = size(states , 2 );
            log_detect = log(sensormodel.P_D / sensormodel.intensity_c);
            log_miss  = log(1 - sensormodel.P_D);
             M = obj.reduction.M;


            for k = 1 : totalTrackTime
                % Gating and cost matrix
                z_k = Z{k};
                mk = size( z_k , 2 );
                L  =  inf(num_objects , num_objects + mk );
                admitted = zeros(num_objects , mk );
                for i = 1 : num_objects
                    for j = 1 : mk 
                    [~, index] = obj.density.ellipsoidalGating( states(i) , z_k(:,j), measmodel, obj.gating.size);
                    admitted(i,j) = index;
                    if index
                             H_ih = measmodel.H(states(i).x);
                             S_ih = H_ih*states(i).P*H_ih' + measmodel.R;
                             inn_ih = z_k(:,j) - measmodel.h(states(i).x);
                             lij = -log_detect ;
                             lij = lij + 0.5*log(det(2*pi*S_ih));
                             lij = lij + 0.5*((inn_ih)')*inv(S_ih)*(inn_ih);
                             L(i,j) = lij;
                        end
                    end
                    L( i , mk + i ) = log_miss;
                end
                % if the column sum of inf elements is == to num objects
                % its full of inf values 
                noninf_cols = sum(isinf(L)) < num_objects;
                L = L(:, noninf_cols);
                z_k = z_k(:, noninf_cols(1:mk));
                mk = size(z_k,2) ; 
                %%%%%%%%%%%%%%%%

                % Find M assignments i e data associations
                [theta, ~, ~]= kBest2DAssign(L, M);
                %%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Normalize log weights. There might not be as hypotheses
                % as data associations. Why ? 
                M = size(theta , 2) ;
                log_w = zeros(M , 1 );
                for iM = 1 : M
                    % sum over the every row ( objects ) and the specific
                    % data association column
                    tr_AL = sum(L(sub2ind(size(L),1:num_objects, theta(:,iM)')));
                    log_w(iM) = -tr_AL;
                end
                %give a value to the misdetection
                theta(theta>mk) = mk + 1 ;
                log_w = normalizeLogWeights(log_w) ; 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % 5. prune assignment matrices that correspond to data association
                % hypotheses with low weights and renormalise the weights
                
                % hypo index
                hyp = 1 : M ; 
                [log_w, hyp] = hypothesisReduction.prune( log_w, hyp, obj.reduction.w_min );
                theta = theta(:,hyp);
                log_w = normalizeLogWeights(log_w) ; 

                %%%%%%%%%%%%%%%%%%%%%
                % 6. create new local hypotheses for each of the data association results;
                beta = zeros(num_objects,mk+1);   % marg. prob that a object i=1:n is associated to meas. j=0:m
                % since its a marginal of i we need to saturate over the
                % measurements
                
                for i = 1 : num_objects
                    for i_theta = 1 : size(theta , 2 )
                        % take the measurement index
                        j = theta(i,i_theta); % object i is associated to which meas. j in data ass. i_theta
                        beta(i,j) = beta(i,j) + exp(log_w(i_theta));
                    end
                end
                % if we saturate over the objects now ( row sum ) we get 1
                % as every object is either misdetected of identified
                %sum(beta , 2)

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % 7. merge local hypotheses that correspond to the same object by moment matching;  

                for i = 1 : num_objects
                    P_pred = states(i).P;
                    x_pred = states(i).x;

                     H = measmodel.H(x_pred);
                     innCov = H*P_pred*H' + measmodel.R;
                     innCov = 0.5*(innCov + innCov');
                     K = P_pred*H'*inv(innCov);
                    
                     eps_mu = zeros(size(z_k , 1) , 1);
                     eps_cov = zeros( size(z_k , 1 ) , size(z_k , 1) ) ; 

                     for j = 1 : mk
                         eps_ij = z_k(:,j) - measmodel.h(x_pred) ;
                         % weighted average innovation
                         eps_mu = eps_mu  +  beta(i,j)*eps_ij;
                         eps_cov = eps_cov + beta(i,j)*eps_ij*eps_ij';


                     end
                     
                     P_bar_i = P_pred - K*(innCov)*K' ;
                    P_tilde_i = K * ( eps_cov - eps_mu*eps_mu' ) * K' ;

                     states(i).x = x_pred + K * eps_mu ;
                     % covariance increases  ( as it should )
                     states(i).P = beta(i,mk+1)*P_pred + ...
                         (1 - beta(i,mk+1) ) * P_bar_i + ...
                         P_tilde_i ; 
                end
                
                % now i think we should have less hypothesis than when
                % started , but we have multiple copies of them
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % 8. extract object state estimates;

               for i =   1    :   num_objects
                    estimates{k}(:,i) = states(i).x;
               end


               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % 9. predict each local hypothesis. array fun applies the
               % function using each index of states as input to the call
              states = arrayfun(@(s) obj.density.predict(s,motionmodel), states );


         
            end



            
end     

%%%%%%%%%%%%%%%%%%%

    end
end
