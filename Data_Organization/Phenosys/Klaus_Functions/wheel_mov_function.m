function Overall_Movement = wheel_mov_function(digital_word)
% Determining the movement of the wheel in degrees

load('Trials_Sync.mat')  

% The unique variable in TTL signal is the wheel movement (until now)
digital_input_0 = (bitand(digital_word, 2^0) > 0);
digital_input_1 = (bitand(digital_word, 2^1) > 0);
digital_input_2 = (bitand(digital_word, 2^2) > 0);
digital_input_3 = (bitand(digital_word, 2^3) > 0);
digital_input_4 = (bitand(digital_word, 2^4) > 0);
digital_input_5 = (bitand(digital_word, 2^5) > 0);


TTL1=digital_input_3;
TTL2=digital_input_4;
% TTL1 = From the pin A of the rotor encoder // digital_input_3 from the
%        digitalin.dat

% TTL2 = From the pin B of the rotor encoder // digital_input_4 from the
%        digitalin.dat

% Variable initialisation
enconderCPR = 1024;                                                         % Encoder resolution
Position = 0;                                                               % Relative start position

% Position decoding
%%%%%%%%%%% Detecting positive changes
A=diff(TTL1);
IndexA=find(A==1)+1;
Overall_Movement=zeros(length(TTL1),1);
%%%%%%%%%%%
count = 1;
count_trial=1;
for i=1:length(TTL1)
    % Reset Position in each new trial
    if(count_trial<=length(Trials_Sync(:,1)))
        if(i==Trials_Sync(count_trial,15))
            Position=0;
            count_trial=count_trial+1;
        end
    end

    if count<=length(IndexA)
        if i==IndexA(count)
            % Checking the value of the pin B when there is movement in A
            if TTL2(i) == 0  
                Position = Position + 360/enconderCPR;         
            else
                Position = Position - 360/enconderCPR;
            end
            count = count + 1;
        end
     end
     Overall_Movement(i)=Position;
end