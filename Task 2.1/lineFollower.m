function [X, Y] = lineFollower(R, G, B)
%LINEFOLLOWER  7x7 grid steering for red-line-following drone.
%   Crops to 119x119, divides into 49 blocks of 17x17.
%   Drone is at centre block (4,4). Scans ring-1 neighbours in a
%   direction-dependent priority order; falls back to max-count block.
%
%   Inputs:  R, G, B  -- uint8 arrays, 120x160
%   Outputs: X, Y     -- unit vector  (X: up/forward,  Y: right)

    persistent lastDir;
    if isempty(lastDir)
        lastDir = 1;       % 1=N (forward/up)
    end

    %% 1. Centre-crop to 119x119, red mask
    r = double(R(1:119, 21:139));
    g = double(G(1:119, 21:139));
    b = double(B(1:119, 21:139));
    mask = double((r > 180) & (g < 100) & (b < 100));

    if sum(sum(mask)) < 20
        X = 1.0;  Y = 0.0;  return;
    end

    %% 2. 7x7 grid of 17x17 blocks — count red pixels
    counts = zeros(7, 7);
    for i = 1:7
        r1 = (i - 1) * 17 + 1;
        for j = 1:7
            c1 = (j - 1) * 17 + 1;
            counts(i, j) = sum(sum(mask(r1:r1+16, c1:c1+16)));
        end
    end

    %% 3. Ring-1 scan: 8 neighbours of centre (4,4)
    %   Compass: N=1  NE=2  E=3  SE=4  S=5  SW=6  W=7  NW=8
    r1R = [3, 3, 4, 5, 5, 5, 4, 3];   % row index per compass dir
    r1C = [4, 5, 5, 5, 4, 3, 3, 3];   % col index per compass dir

    % Only scan forward cone (±90°): forward, ±45°, ±90°
    % Excludes ±135° and behind — prevents reversals
    baseOff   = [0, 1, 7, 2, 6];
    scanOrder = mod(lastDir - 1 + baseOff, 8) + 1;

    THRESH = 0.10 * 17 * 17;          % ~29 red pixels => block active

    chosenR = 0;
    chosenC = 0;
    found   = false;

    % First block in priority order that exceeds threshold wins
    for p = 1:5
        si = scanOrder(p);
        if counts(r1R(si), r1C(si)) >= THRESH
            chosenR = r1R(si);
            chosenC = r1C(si);
            found   = true;
            break;
        end
    end

    % If no forward-cone block hit threshold, pick the one with most red
    % among the same 5 forward-cone directions only
    if ~found
        maxCnt = 0.0;
        for p = 1:5
            si = scanOrder(p);
            cnt = counts(r1R(si), r1C(si));
            if cnt > maxCnt
                maxCnt  = cnt;
                chosenR = r1R(si);
                chosenC = r1C(si);
                found   = true;
            end
        end
    end

    %% 4. Fallback: highest-count non-centre block (forward half only)
    if ~found
        maxCnt = 0.0;
        for i = 1:7
            for j = 1:7
                if (i == 4) && (j == 4)
                    continue;
                end
                % Skip blocks that are behind the drone
                dr = 4 - i;
                dc = j - 4;
                compassDir = mod(round(atan2(double(dc), double(dr)) / (pi/4)), 8) + 1;
                % Check angular distance from lastDir
                angDist = mod(compassDir - lastDir + 4, 8) - 4; % range -4..3
                if abs(angDist) > 2
                    continue;           % skip blocks > 90° from forward
                end
                if counts(i, j) > maxCnt
                    maxCnt  = counts(i, j);
                    chosenR = i;
                    chosenC = j;
                end
            end
        end
        if maxCnt >= 1
            found = true;
        end
    end

    % Last resort: no red in forward half at all — keep going straight
    if ~found
        X = 1.0;  Y = 0.0;  return;
    end

    %% 5. Steer toward chosen block centre
    bRow = (chosenR - 1) * 17 + 9;    % block-centre row in cropped image
    bCol = (chosenC - 1) * 17 + 9;    % block-centre col in cropped image

    steerX = 60.0 - bRow;             % positive X = up = forward
    steerY = bCol - 60.0;             % positive Y = right

    mag = sqrt(steerX^2 + steerY^2);
    if mag < 1e-9
        X = 1.0;  Y = 0.0;
    else
        X = steerX / mag;
        Y = steerY / mag;
    end

    %% 6. Update lastDir for next frame
    dr = 4 - chosenR;                  % positive = north (up)
    dc = chosenC - 4;                  % positive = east  (right)
    lastDir = mod(round(atan2(double(dc), double(dr)) / (pi/4)), 8) + 1;
end