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
            Hx = measmodel.H(state_pred.x);
            %Innovation covariance
            S = Hx*state_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Calculate predicted likelihood
            predicted_likelihood = log_mvnpdf(z',measmodel.h(state_pred.x)',S);

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
            if isempty(z)
                z_ingate = [];
                meas_in_gate = [];
            else
            
                M = size(z,2); %Number of measurements
                meas_in_gate = false(M,1);

                Hx = measmodel.H(state_pred.x);
                R = measmodel.R;
                z_bar = Hx*state_pred.x;
                S = Hx*state_pred.P*Hx' + R; 
                S = (S + S')/2;
                S_inv = inv(S);

                for i=1:M
                    nu = z(:,i) - z_bar;
                    dMahal = nu'*S_inv*nu;
                    if dMahal < gating_size
                        meas_in_gate(i) = true;
                    end
                end

                ingate_indexes = find(meas_in_gate==1);
                z_ingate = z(:,ingate_indexes);
            end
            
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
            N = length(w);
            n = size(states(1).x,1);
            w_merged = sum(w);
            mu_merged = zeros(n,1);
            for i=1:N
                mu_merged = mu_merged + w(i)*states(i).x;
            end
            mu_merged = mu_merged/w_merged;
            
            Sigma_merged = zeros(n,n);
            for i=1:N
                Sigma_merged = Sigma_merged + (w(i)/w_merged)*(states(i).P + (states(i).x - mu_merged)*(states(i).x - mu_merged)');
            end
            
            
            
            state.x = mu_merged;
            state.P = Sigma_merged;
            
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
        
        function [w_hat,states_hat] = RunnalsMRA(w,states,threshold)
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
            
            w = exp(w);
            BMatrix = zeros(length(w),length(w));

            while(length(w)-threshold>0)
                %We first compute the KLD bounds for every merging action
                for i=1:length(states)
                    for j=1:length(states)
                        if(i<j)
                            BMatrix(i,j) = KLDBound([w(i);w(j)],[states(i);states(j)],GaussianDensity);
                        end
                    end
                end
                %We then find the action with the lowest KLD bound and we merge the
                %corresponding mixture components
                [i,j] = find(BMatrix == min(BMatrix(BMatrix>0)));
                pdf_merged = GaussianDensity.momentMatching([w(i);w(j)],[states(i);states(j)]);
                states(i) = pdf_merged;
                w(i) = w(i) + w(j);
                states(j) = [];
                w(j) = [];
                %We then shrink both the component vector and bound matrix
                BMatrix = BMatrix(1:end-1,1:end-1);

            end
            
            w_hat = log(w);
            states_hat = states;
        end 
        
    end
end