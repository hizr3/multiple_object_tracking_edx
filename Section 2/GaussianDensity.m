%Model structures need to be called:
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
classdef GaussianDensity
    
    methods (Static)
        
        function expected_value = expectedValue(state)
            expected_value = state.x;
        end
        
        function covariance = covariance(state)
            covariance = state.P;
        end
        
        function state_pred = predict(state, motionmodel)
            %PREDICT performs linear/nonlinear (Extended) Kalman prediction
            %step 
            %INPUT: state: a structure with two fields:
            %                   x: object state mean --- (state dimension)
            %                   x 1 vector 
            %                   P: object state covariance --- (state
            %                   dimension) x (state dimension) matrix 
            %       motionmodel: a structure specifies the motion model
            %       parameters 
            %OUTPUT:state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state
            %                   dimension) x 1 vector 
            %                   P: predicted object state covariance ---
            %                   (state dimension) x (state dimension)
            %                   matrix  
            
            state_pred.x = motionmodel.f(state.x);
            state_pred.P = motionmodel.F(state.x)*state.P*motionmodel.F(state.x)'+motionmodel.Q;
            
        end
        
        function state_upd = update(state_pred, z, measmodel)
            %UPDATE performs linear/nonlinear (Extended) Kalman update step
            %INPUT: z: measurements --- (measurement dimension) x 1 vector
            %       state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state
            %                   dimension) x 1 vector 
            %                   P: predicted object state covariance ---
            %                   (state dimension) x (state dimension)
            %                   matrix  
            %       measmodel: a structure specifies the measurement model
            %       parameters 
            %OUTPUT:state_upd: a structure with two fields:
            %                   x: updated object state mean --- (state
            %                   dimension) x 1 vector 
            %                   P: updated object state covariance ---
            %                   (state dimension) x (state dimension)
            %                   matrix  
            
            %Measurement model Jacobian
            Hx = measmodel.H(state_pred.x);
            %Innovation covariance
            S = Hx*state_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            
            K = (state_pred.P*Hx')/S;
            
            %State update
            state_upd.x = state_pred.x + K*(z - measmodel.h(state_pred.x));
            %Covariance update
            state_upd.P = (eye(size(state_pred.x,1)) - K*Hx)*state_pred.P;
            
        end
        
        function predicted_likelihood = predictedLikelihood(state_pred,z,measmodel)
            %PREDICTLIKELIHOOD calculates the predicted likelihood in
            %logarithm domain 
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %        of measurements) matrix 
            %        state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state
            %                   dimension) x 1 vector 
            %                   P: predicted object state covariance ---
            %                   (state dimension) x (state dimension)
            %                   matrix  
            %        measmodel: a structure specifies the measurement model
            %        parameters 
            %OUTPUT: predicted_likelihood: predicted likelihood for
            %        each measurement in logarithmic scale --- (number of
            %        measurements) x 1 vector         

            Hx = measmodel.H(state_pred.x)
              %Innovation covariance
            S = Hx*state_pred.P*Hx' + measmodel.R
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            % object measurement prediction
            zk = measmodel.h(state_pred.x)
            predicted_likelihood = log(mvnrnd(z' - zk',S))
            
        end
        
        function [z_ingate, meas_in_gate] = ellipsoidalGating(state_pred, z, measmodel, gating_size)
            %ELLIPSOIDALGATING performs ellipsoidal gating for a single
            %object 
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %        of measurements) matrix 
            %        state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state
            %                   dimension) x 1 vector 
            %                   P: predicted object state covariance ---
            %                   (state dimension) x (state dimension)
            %                   matrix  
            %        measmodel: a structure specifies the measurement model
            %        parameters 
            %        gating_size: gating size --- scalar
            %OUTPUT: z_ingate: measurements in the gate --- (measurement
            %        dimension) x (number of measurements in the gate)
            %        matrix
            %        meas_in_gate: boolean vector indicating whether the
            %        corresponding measurement is in the gate or not ---
            %        (number of measurements) x 1
        
            
            %%%%%%% SOLUZIONE MIA %%%%%%%%%%%%%%
            
            
        % Squared Mahalanobis distance
            sq_md = @(x,y,S) (x-y)'*inv(S)*(x-y);
            
            % Allocate space for needed data
            num_meas = size(z,2);
            meas_in_gate = zeros(num_meas,1);

            % We do not know a priori how many measurements will be in the gate, 
            % so i'd rather allocate the maximum space and truncate the matrix
            % using a counter, at the end of the cycle

            z_ingate = z ;
            
            Hx = measmodel.H(state_pred.x);
            z_pred = measmodel.H(state_pred.x) ; 
            S = Hx*state_pred.P*Hx' + measmodel.R ; 
            S = ( S + S' )*0.5 ;

            k = 0;
            for i = 1 : num_meas
                % Index i selects one measurement
                if sq_md(z_pred,z(:,i),S) < gating_size
                    meas_in_gate(i) = 1 ;
                    k = k + 1 ; 
                    z_ingate(:,i) = z(:,k);
                end         
            end
            % disp(meas_in_gate);
            % disp(z_ingate)

            % Convert to boolean
            meas_in_gate_mine = logical(meas_in_gate);
            %Truncate
            z_ingate_mine = z_ingate(:,1:k);
              
            %%%%%%% SOLUZIONE MIA %%%%%%%%%%%%%%
            
            
        % Squared Mahalanobis distance
            sq_md = @(x,y,S) (x-y)'*inv(S)*(x-y);
            
            % Allocate space for needed data
            num_meas = size(z,2);
            meas_in_gate = zeros(num_meas,1);

            % We do not know a priori how many measurements will be in the gate, 
            % so i'd rather allocate the maximum space and truncate the matrix
            % using a counter, at the end of the cycle

            z_ingate = z ;
            
            Hx = measmodel.H(state_pred.x);
            z_pred = measmodel.H(state_pred.x) ; 
            S = Hx*state_pred.P*Hx' + measmodel.R ; 
            S = ( S + S' )*0.5 ;

            k = 0;
            for i = 1 : num_meas
                % Index i selects one measurement
                if sq_md(z_pred,z(:,i),S) < gating_size
                    meas_in_gate(i) = 1 ;
                    k = k + 1 ; 
                    z_ingate(:,i) = z(:,k);
                end         
            end
            % disp(meas_in_gate);
            % disp(z_ingate)

            % Convert to boolean
            meas_in_gate = logical(meas_in_gate);
            %Truncate
            z_ingate = z_ingate(:,1:k);
        
            % Convert to boolean
            meas_in_gate_mine = logical(meas_in_gate);
            %Truncate
            z_ingate_mine = z_ingate(:,1:k);
            
            % %%%%%%%%%%%%%%% SOLUZIONE GITHUB %%%%%%%%%%%%%%%%
            % 
                meas_in_gate = zeros(size(z,2),1);
             for i=1:size(state_pred.x,2)
                 % preicted measurement covariance
                 Hx = measmodel.H(state_pred.x(:,i));
                 %Innovation covariance
                 S = Hx * state_pred.P(:,:,i) * Hx' + measmodel.R;
                 % ensure it is positive definite
                 S = (S+S')/2;
                 % object measurement prediction
                 zk = measmodel.h(state_pred.x(:,i));
                meas_in_gate = meas_in_gate | diag( (zk-z)'*inv(S)*(zk-z)) < gating_size;
            end
             z_ingate = z(:,meas_in_gate);
            %%%%%%%%%%%%%%%%%%%%%%% CHE CAMBIA ???? %%%%%%%%%%%%%%%%%%%%%%
            
           % check = isequal(z_ingate,z_ingate_mine)
           % check2 = isequal(meas_in_gate,meas_in_gate_mine)
            
            
        end
        
        function state = momentMatching(w, states)
            %MOMENTMATCHING: approximate a Gaussian mixture density as a
            %single Gaussian using moment matching 
            %INPUT: w: normalised weight of Gaussian components in
            %       logarithm domain --- (number of Gaussians) x 1 vector 
            %       states: structure array of size (number of Gaussian
            %       components x 1), each structure has two fields 
            %               x: means of Gaussian components --- (variable
            %               dimension) x 1 vector 
            %               P: variances of Gaussian components ---
            %               (variable dimension) x (variable dimension) matrix  
            %OUTPUT:state: a structure with two fields:
            %               x: approximated mean --- (variable dimension) x
            %               1 vector 
            %               P: approximated covariance --- (variable
            %               dimension) x (variable dimension) matrix 
             
            if length(w) == 1
                state = states;
                return;
            end
   
            w = exp(w);
            mk = size(w,1);
            dim_state = size(states(1).x, 1 )
                        
            state.x = zeros(dim_state,1);
            state.P = zeros(dim_state);
            for i = 1 : mk
                state.x = state.x + w(i)*states(i).x;
            end
             for i = 1 : mk
                state.P = state.P + w(i)*states(i).P + w(i)*(states(i).x-state.x)*(states(i).x-state.x)';
             end
             
           
        end
        
        function [w_hat,states_hat] = mixtureReduction(w,states,threshold)
            %MIXTUREREDUCTION: uses a greedy merging method to reduce the
            %number of Gaussian components for a Gaussian mixture density 
            %INPUT: w: normalised weight of Gaussian components in
            %       logarithmic scale --- (number of Gaussians) x 1 vector 
            %       states: structure array of size (number of Gaussian
            %       components x 1), each structure has two fields 
            %               x: means of Gaussian components --- (variable
            %               dimension) x (number of Gaussians) matrix 
            %               P: variances of Gaussian components ---
            %               (variable dimension) x (variable dimension) x
            %               (number of Gaussians) matrix  
            %       threshold: merging threshold --- scalar
            %INPUT: w_hat: normalised weight of Gaussian components in
            %       logarithmic scale after merging --- (number of
            %       Gaussians) x 1 vector  
            %       states_hat: structure array of size (number of Gaussian
            %       components after merging x 1), each structure has two
            %       fields  
            %               x: means of Gaussian components --- (variable
            %               dimension) x (number of Gaussians after
            %               merging) matrix  
            %               P: variances of Gaussian components ---
            %               (variable dimension) x (variable dimension) x
            %               (number of Gaussians after merging) matrix  
            
            if length(w) == 1
                w_hat = w;
                states_hat = states;
                return;
            end
            
            %Index set of components
            I = 1:length(states);
            el = 1;
            
            while ~isempty(I)
                Ij = [];
                %Find the component with the highest weight
                [~,j] = max(w);

                for i = I
                    temp = states(i).x-states(j).x;
                    val = diag(temp.'*(states(j).P\temp));
                    %Find other similar components in the sense of small
                    %Mahalanobis distance 
                    if val < threshold
                        Ij= [ Ij i ];
                    end
                end
                
                %Merge components by moment matching
                [temp,w_hat(el,1)] = normalizeLogWeights(w(Ij));
                states_hat(el,1) = GaussianDensity.momentMatching(temp, states(Ij));
                
                %Remove indices of merged components from index set
                I = setdiff(I,Ij);
                %Set a negative to make sure this component won't be
                %selected again 
                w(Ij,1) = log(eps);
                el = el+1;
            end
            
        end
        
    end
end