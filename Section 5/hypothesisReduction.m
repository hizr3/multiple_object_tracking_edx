classdef hypothesisReduction
    %HYPOTHESISREDUCTION is class containing different hypotheses reduction
    %method 
    %PRUNE: prune hypotheses with small weights.
    %CAP:   keep M hypotheses with the highest weights and discard the rest. 
    %MERGE: merge similar hypotheses in the sense of small Mahalanobis
    %       distance.
    
    methods (Static)
        function [hypothesesWeight, multiHypotheses] = ...
                prune(hypothesesWeight, multiHypotheses, threshold)
            %PRUNE prunes hypotheses with small weights
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: hypotheses with weights smaller than this
            %       threshold will be discarded --- scalar in logarithmic scale
            %OUTPUT:hypothesesWeight: hypotheses weights after pruning in
            %       logarithmic scale --- (number of hypotheses after
            %       pruning) x 1 vector   
            %       multiHypotheses: (number of hypotheses after pruning) x
            %       1 structure  
            N = length(hypothesesWeight);
            goodHypotheses = false(N,1);
            for i=1:N
                if hypothesesWeight(i) >= threshold
                    goodHypotheses(i) = true;
                end
            end
            indexes = find(goodHypotheses == 1);
            hypothesesWeight = hypothesesWeight(indexes);
            multiHypotheses = multiHypotheses(indexes);
            
        end
        
        function [hypothesesWeight, multiHypotheses] = ...
                cap(hypothesesWeight, multiHypotheses, M)
            %CAP keeps M hypotheses with the highest weights and discard
            %the rest 
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector  
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       M: only keep M hypotheses --- scalar
            %OUTPUT:hypothesesWeight: hypotheses weights after capping in
            %       logarithmic scale --- (number of hypotheses after
            %       capping) x 1 vector 
            %       multiHypotheses: (number of hypotheses after capping) x
            %       1 structure 
            if length(hypothesesWeight)>M
                [hypothesesWeight,indexes] = sort(hypothesesWeight,'descend');
                multiHypotheses = multiHypotheses(indexes);
                hypothesesWeight = hypothesesWeight(1:M);
                multiHypotheses = multiHypotheses(1:M);
            end
           
        end
        
        function [hypothesesWeight,multiHypotheses] = ...
                merge(hypothesesWeight,multiHypotheses,threshold,density)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector  
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: merging threshold --- scalar
            %       density: a class handle
            %OUTPUT:hypothesesWeight: hypotheses weights after merging in
            %       logarithmic scale --- (number of hypotheses after
            %       merging) x 1 vector  
            %       multiHypotheses: (number of hypotheses after merging) x
            %       1 structure 
            
            [hypothesesWeight,multiHypotheses] = ...
                density.mixtureReduction(hypothesesWeight,multiHypotheses,threshold);
            
        end
        
        function [hypothesesWeight,multiHypotheses] = ...
                mergeRunnals(hypothesesWeight,multiHypotheses,threshold,density)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %       in logarithmic scale --- (number of hypotheses) x 1
            %       vector  
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: merging threshold --- scalar
            %       density: a class handle
            %OUTPUT:hypothesesWeight: hypotheses weights after merging in
            %       logarithmic scale --- (number of hypotheses after
            %       merging) x 1 vector  
            %       multiHypotheses: (number of hypotheses after merging) x
            %       1 structure 
            
            [hypothesesWeight,multiHypotheses] = ...
                density.RunnalsMRA(hypothesesWeight,multiHypotheses,threshold);
            
        end
        
    end
end