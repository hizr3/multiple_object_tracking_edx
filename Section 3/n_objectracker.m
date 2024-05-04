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
            N = length(states);
            numSteps = length(Z);
            mis_cost = -log(1-sensormodel.P_D);
            estimates = cell(numSteps,1);
            for k=1:numSteps
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
                    L(i,M+i) = mis_cost;
                end
                %Reduction of the cost matrix
                indexes = find(sum(cell2mat(meas_in_gate),2) == 0); %With this command we find the indexes of the measurements not falling in any gate
                L(:,indexes) = [];
                z(:,indexes) = [];
                %Assignment problem solution
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
                
            end
        end
        
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
            N = length(states);
            numSteps = length(Z);
            mis_cost = -log(1-sensormodel.P_D);
            estimates = cell(numSteps,1);
            for k=1:numSteps
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
                    L(i,M+i) = mis_cost;
                end
                %Reduction of the cost matrix
                indexes = find(sum(cell2mat(meas_in_gate),2) == 0); %With this command we find the indexes of the measurements not falling in any gate
                L(:,indexes) = []; % We remove columns in the cost matrix with only Inf
                z(:,indexes) = [];
                %Assignment problem solution
                [col4rowBest,~,gain] = kBest2DAssign(L,obj.reduction.M);
                
                
                %Here we compute the normalised data association probabilities in the
                %log domain
                logDAWeights = -gain;
                logDAWeights = normalizeLogWeights(logDAWeights);
                
                %Prune assignment matrices that correspond to data
                %association hypotheses with low weights and renormalise
                %the weights
                hypIndexes = 1:length(logDAWeights);
                [logDAWeights, hypIndexes] = hypothesisReduction.prune(logDAWeights, hypIndexes, obj.reduction.w_min);
                logDAWeights = normalizeLogWeights(logDAWeights);
                col4rowBest = col4rowBest(:,hypIndexes);
              
                %We now create new local hypotheses for each of the data
                %association results. We compute approximated marginal
                %probabilities because in the moment matching phase we will
                %need the weights for every local hypothesis
                betaij = zeros(N,size(z,2));
                for i=1:N
                    for j=1:size(z,2)
                        indexes = find(col4rowBest(i,:)==j);
                        betaij(i,j) = sum(exp(logDAWeights(indexes)));
                    end
                end
                
                betai0 = 1-sum(betaij,2);
 
                %We can now create new hypotheses for each association
                %result
                for i=1:N
                    llh = [];
                    hypotheses = [];
                    thetai = unique(col4rowBest(i,:));
                    %Take only detection indexes, we manage the
                    %misdetection component separately
                    ind = find(thetai<=size(z,2)); %Every index above the number of measurements is a misdetection
                    thetai= thetai(ind);
                    if betai0(i) > 0            %The betai0 weight can be zero, in that case we get -Inf as weight, which is problematic
                        llh = [llh;log(betai0(i))]; %On the other hand, if betai0=0 thetai should not be empty
                        hypotheses = [hypotheses;states(i)];
                    end
 
                    if ~isempty(thetai)
                        for j=1:length(thetai)
                            hypotheses = [hypotheses; GaussianDensity.update(states(i),z(:,thetai(j)),measmodel)];
                        end
                        llh = [llh;log(betaij(i,thetai))'];
                        %We do a merging of local hypotheses which
                        %minimizes the KLD of the resulting approximation
                        states(i) = GaussianDensity.momentMatching(llh,hypotheses);
                    end
                    %Note: if betai0 is zero it means that we get at least one
                    %really strong detection. It cannot happen that betai0=0
                    %and thetai is empty

                end
                %We extract object estimates
                estimates{k} = [states.x];
                %Prediction
                states = arrayfun(@(state) GaussianDensity.predict(state,motionmodel), states);
            end
        end
        
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
            N = length(states);
            numSteps = length(Z);
            md_llh = log(1-sensormodel.P_D);
            H_prev = ones(1,N); %Initial LH Table; only one hypothesis per object
            llh_prev = 0; %log(1), the global hypothesis is only one
            localHyps_prev = cell(N,1); %Local hypotheses Hk-1_i(one hyp per object)
            %This time it is necessary to group hypotheses for each object
            %in separate groups. Using cells might be a good data structure
            %to face this task
            for i=1:N
                localHyps_prev{i} = states(i);
            end
            estimates = cell(numSteps,1);
            
            for k=1:numSteps
                z = cell2mat(Z(k));
                M = size(z,2);
                z_ingate = cell(N,1);
                meas_in_gate = cell(N,1);
                meas_occurrences = zeros(1,M);
                %For each local hypothesis in each hypothesis tree
                %Implement ellipsoidal gating
                for i=1:N
                    [z_ingate{i}, meas_in_gate{i}] = arrayfun(@(hyp) GaussianDensity.ellipsoidalGating(hyp,z,measmodel,obj.gating.size),localHyps_prev{i},'UniformOutput',false);
                    meas_occurrences = meas_occurrences + sum([meas_in_gate{i}{:}],2)';
                end
                %I preferred to perform the gating at first, in order to be
                %able to filter out all the measurements not falling in any
                %gate. This simplifies alot the indexing phase later in the
                %code.
                indexes = find(meas_occurrences==0);
                z(:,indexes) = [];
                M = size(z,2);
                llh_i = cell(N,1);
                localHyps = cell(N,1);
                %Calculate missed detection and predicted likelihood for
                %each measurement inside the gate
                for i=1:N
                    Hk_i = length(localHyps_prev{i});
                    %We know that the number of updated hypotheses for each
                    %object is Hk_i*(mk+1) at time k. We conside M in the indexing
                    %instead of the gated measurements size because it allows us to
                    %use directly (or almost) the result from the kBest
                    %assignment. To do so we initialize the logweights of
                    %the hk_ith hypothesis as following
                    llh_i{i} = -Inf(Hk_i*(M+1),1);
                    for hk_i=1:Hk_i
                        meas_in_gate{i}{hk_i}(indexes) = [];
                        %Even if we consider M hypotheses here, we only
                        %compute the update and the llh for the ones inside
                        %the gate
                        in_gate_indexes = find(meas_in_gate{i}{hk_i}==1)';
                        for j=in_gate_indexes
                            index = (hk_i-1)*(M+1)+j;
                            llh_i{i}(index) = log(sensormodel.P_D) + GaussianDensity.predictedLikelihood(localHyps_prev{i}(hk_i),z(:,j),measmodel)...
                                - log(sensormodel.intensity_c);
                            localHyps{i}(index) = GaussianDensity.update(localHyps_prev{i}(hk_i),z(:,j),measmodel);

                        end
                        %We compute here llh and a new hypothesis for the
                        %misdetection case
                        index = (hk_i-1)*(M+1)+M+1;
                        llh_i{i}(index) = md_llh;
                        localHyps{i}(index) = localHyps_prev{i}(hk_i);
                    end
                end
                
                H = [];
                llh = [];
                %For each predicted global hypothesis
                for hk=1:size(H_prev,1)
                    %We create the cost matrix
                    L = Inf(N,M+N);
                    
                    for i=1:N
                        startIndex = (H_prev(hk,i)-1)*(M+1)+1;
                        endIndex = startIndex + M - 1;
                        L(i,1:M) = -llh_i{i}(startIndex:endIndex)';
                        L(i,M+i) = -llh_i{i}(endIndex+1);
                    end
                    %We obtain the M best assignments using the solver
                    [col4rowBest, ~, gain] = kBest2DAssign(L,ceil(exp(llh_prev(hk))*obj.reduction.M));
                    
                    %Considering that the misdetection costs are standing
                    %on the diagonal of the right submatrix, their indexes
                    %will not be suitable directly for the local
                    %hypotheses. Given that we took the precaution of
                    %considering M instead of the gate measurements size,
                    %it will be necessary just to fix the i-th misdetection cost
                    %to the column M+1
                    for i=1:N
                        col4rowBest(i,col4rowBest(i,:)>M) = M+1;
                    end
                    
                    %Having now the best assignments, we can update the
                    %global LUT
                    G = length(gain);
                    
                    for g=1:G
                        %Update of the logweights
                        llh(end+1,1) = llh_prev(hk) - gain(g);
                        %Update of the LH Table
                        H(end+1,:) = zeros(1,N);
                        for i=1:N
                            H(end,i) = (H_prev(hk,i)-1)*(M+1) + col4rowBest(i,g);
                        end
                    end

                end
                
                %Normalise global hypothesis weights and perform pruning
                %and capping
                llh = normalizeLogWeights(llh);
                [llh, indexes] = hypothesisReduction.prune(llh,1:length(llh),obj.reduction.w_min);
                llh = normalizeLogWeights(llh);
                H = H(indexes,:);
                [llh, indexes] = hypothesisReduction.cap(llh,1:length(llh),obj.reduction.M);
                llh = normalizeLogWeights(llh);
                H = H(indexes,:);
                
                %Prune the local hypotheses not falling in any of the
                %global hypotheses
                for i=1:N
                    hk_i = unique(H(:,i));
                    localHyps{i} = localHyps{i}(hk_i);
                    llh_i{i} = llh_i{i}(hk_i);
                    llh_i{i} = normalizeLogWeights(llh_i{i});
                    %Reindex the global hypothesis LUT
                    for j=1:length(hk_i)
                        H(H(:,i)==hk_i(j),i) = j;
                    end
                end
                
                %Exctract object state estimates from the global hypothesis
                %with the highest weight
                [~,llh_star] = max(llh);
                H_star = H(llh_star,:);
                
                for i=1:N
                    estimates{k}(:,i) = localHyps{i}(H_star(i)).x;
                end
                
                
                %We predict each local hypothesis in each hypothesis tree
                for i=1:N
                    localHyps{i} = arrayfun(@(hyp) GaussianDensity.predict(hyp,motionmodel), localHyps{i});
                end
                
                %We set the actual hypotheses, LUT and global weights to be
                %the old ones for the next iteration
                localHyps_prev = localHyps;
                H_prev = H;
                llh_prev = llh;
                
            end
        end
    end
end
