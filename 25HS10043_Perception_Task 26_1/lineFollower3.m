function [X, Y] = lineFollower3(R, G, B)
%LINEFOLLOWER3  Line follower using ahead/perpendicular centroid blending.
%
%   Detects red pixels, computes a heading-weighted "ahead" centroid and a
%   perpendicular centroid, blends them adaptively, and steers toward the
%   result.  A persistent heading provides frame-to-frame continuity.
%
%   Direction-independent.  All arrays are fixed-size (Simulink safe).
%
%   Inputs :  R, G, B  — uint8, 120x160
%   Outputs:  X, Y     — unit vector (X = up in image, Y = right)

    % ---- Persistent state ----
    persistent dirX dirY;
    if isempty(dirX)
        dirX = 1.0;  dirY = 0.0;
    end

    %% 1. Red mask  (120x160, binary)
    mask = double((double(R) > 180) & (double(G) < 100) & (double(B) < 100));
    totalRed = sum(sum(mask));

    % Too few red pixels — maintain last heading
    if totalRed < 15
        X = dirX;  Y = dirY;  return;
    end

    %% 2. Global centroid of all red pixels
    colIdx = 1:160;                                    % 1x160 (fixed)
    rowIdx = (1:120)';                                 % 120x1 (fixed)
    gcCol  = sum(sum(mask, 1) .* colIdx) / totalRed;   % column centroid
    gcRow  = sum(sum(mask, 2) .* rowIdx) / totalRed;   % row centroid

    imgCR = 60.0;   % image centre row
    imgCC = 80.0;   % image centre col

    %% 3. Ahead centroid — red pixels in the forward half (heading-relative)
    %   proj(i,j) = (imgCR-i)*dirX + (j-imgCC)*dirY   (positive = ahead)
    %   Weight = max(0, proj+1) so side-pixels contribute slightly.
    projRow = (imgCR - rowIdx) * dirX;                 % 120x1
    projCol = (colIdx - imgCC) * dirY;                 % 1x160
    aheadWt  = mask .* max(0, projRow + projCol + 1.0);
    aheadTot = sum(sum(aheadWt));

    if aheadTot > 1.0
        aheadCol = sum(sum(aheadWt, 1) .* colIdx) / aheadTot;
        aheadRow = sum(sum(aheadWt, 2) .* rowIdx) / aheadTot;
    else
        aheadCol = gcCol;  aheadRow = gcRow;           % fallback
    end

    %% 4. Perpendicular centroid — emphasises pixels far from heading axis
    %   perpDist(i,j) = (imgCR-i)*(-dirY) + (j-imgCC)*dirX
    %   Weight = |perpDist|  →  new-branch pixels dominate at corners.
    perpRow = (imgCR - rowIdx) * (-dirY);              % 120x1
    perpCol = (colIdx - imgCC) * dirX;                 % 1x160
    perpWt  = mask .* abs(perpRow + perpCol);
    perpTot = sum(sum(perpWt));

    if perpTot > 1.0
        perpCol_c = sum(sum(perpWt, 1) .* colIdx) / perpTot;
        perpRow_c = sum(sum(perpWt, 2) .* rowIdx) / perpTot;
    else
        perpCol_c = gcCol;  perpRow_c = gcRow;         % fallback
    end

    %% 5. Blend: ahead dominates on straights, perp takes over at corners
    %   behindRatio ~ 0 on straights, ~ 0.5–0.8 at sharp corners.
    behindRatio = 1.0 - min(aheadTot / max(totalRed, 1.0), 1.0);
    blendCol = (1.0 - behindRatio) * aheadCol + behindRatio * perpCol_c;
    blendRow = (1.0 - behindRatio) * aheadRow + behindRatio * perpRow_c;

    %% 6. Steer from image centre toward blended target
    steerX = imgCR - blendRow;          % positive = up (forward)
    steerY = blendCol - imgCC;          % positive = right

    % Normalise to unit vector
    mag = sqrt(steerX^2 + steerY^2);
    if mag > 1e-9
        X = steerX / mag;
        Y = steerY / mag;
    else
        X = dirX;  Y = dirY;           % fallback: keep heading
    end

    %% 7. Smooth-update persistent heading  (70% new, 30% old)
    newDirX = 0.7 * X + 0.3 * dirX;
    newDirY = 0.7 * Y + 0.3 * dirY;
    magD = sqrt(newDirX^2 + newDirY^2);
    if magD > 1e-9
        dirX = newDirX / magD;
        dirY = newDirY / magD;
    end
end
