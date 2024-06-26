classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
           % Predict Gaussian component in the Poisson intensity for pre-existing objects.
           old_weights = obj.paras.w;
           %disp(old_weights)
           Hk_old = size(old_weights , 1 ) ;  
           old_hypo = obj.paras.states;
           
           Hk_b = size(birthmodel , 2 ) ; 
           new_hypo = rmfield(birthmodel,'w')' ; %transpose. this removes the weight field
           
           for h = 1 : Hk_old
               old_weights(h) = log(P_S)  + old_weights(h);
               old_hypo(h) = obj.density.predict(old_hypo(h),motionmodel);  
           end
           
           new_weights = zeros(Hk_b , 1 );
           for i = 1 : Hk_b
               new_weights(i) = birthmodel(i).w ;
           end
        
           obj.paras.states = [ new_hypo ; old_hypo ];
           obj.paras.w = [ new_weights ; old_weights ] ;
           
           

        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
          
            Hk_prev = length(obj.paras.w);
            
            llh_md = log(1-P_D) + obj.paras.w;
            hyps_md = obj.paras.states;
            
            %Ellipsoidal gating
            meas_in_gate = cell(Hk_prev,1);
            for hk=1:Hk_prev
                [~,meas_in_gate{hk}] = obj.density.ellipsoidalGating(hyps_md(hk), z, measmodel, gating.size);
            end
            M = reshape(cell2mat(meas_in_gate), size(z,2), Hk_prev);
            indexes = find(sum(M,2)>0); %We find the measurement indexes associated with no gates
            z = z(:,indexes); %We reduce the measurement vector accordingly
            M = M(indexes,:);   
            mk = size(z,2);
            Hk = Hk_prev*(mk+1); %We allocate the full updated hypotheses vectors
            hyps(Hk,1) = struct('x',[],'P',[]);
            llh = -Inf(Hk,1);
            hyps(1:Hk_prev) = hyps_md;
            llh(1:Hk_prev) = llh_md;
            %Note: only the elements corresponding to a gated measurement
            %will be computed. Nevertheless, by allocating the full vectors
            %we can use the theoretically presented indexing
            
            for i=1:mk %We loop over the reduced measurement vector
                normFactor = 0;
                hk_ind = find(M(i,:)>0); %We find the indexes of the hypotheses having the ith measurement in their gates
                for hk=hk_ind
                    hyps(i*Hk_prev+hk) = obj.density.update(hyps(hk), z(:,i), measmodel);
                    llh(i*Hk_prev+hk) = log(P_D) + obj.paras.w(hk) + obj.density.predictedLikelihood(hyps(hk), z(:,i), measmodel);
                    normFactor = normFactor + exp(llh(i*Hk_prev+hk));
                end                         
                %At this point we normalize the bernoulli components in
                %order to be sure that their existence probabilities are at
                %most 1 (at most one potential object associated)
                normFactor = log(normFactor + intensity_c);
                for hk=hk_ind
                    llh(i*Hk_prev+hk) = llh(i*Hk_prev+hk) - normFactor;
                end
            end
            
            %We shrink the vector to remove uncomputed elements
            indexes = find(llh>-Inf);
            llh = llh(indexes);
            hyps = hyps(indexes);
            
                obj.paras.w = llh;
                obj.paras.states = hyps;

        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            
           
             % [ curr_ws ,  curr_hypo  ] = componentReduction( obj , reduction );
             
             %Get a mean estimate of the cardinality of objects by taking the summation of the weights of the Gaussian components  (rounded to the nearest integer), denoted as n.
              n_k = round(sum(  exp(obj.paras.w) ));
              n_k = min(n_k , length( obj.paras.w ) ) ; 
              [out , admitted ] = sort ( obj.paras.w  , 'descend' )  ;
              
              %Extract n object states from the means of the n Gaussian components with the highest weights.
                state_dim = size( obj.paras.states(1).x  , 1 ) ; 
            estimates = zeros( state_dim  ,  n_k);
            
            for i = 1 : n_k
                estimates(:,i) = obj.paras.states(admitted(i)).x ; 
            end

        end
        
    end
    
end

