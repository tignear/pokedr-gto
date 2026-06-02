#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringError {
    InvalidRank { rank: u8 },
    NotEnoughOpponentRatings,
}

pub const PLAYER_COUNT: usize = 6;

const RANK_POINTS: [i32; PLAYER_COUNT] = [35, 21, 7, -7, -21, -35];

pub fn rank_points(rank: u8) -> Result<i32, ScoringError> {
    let index = rank
        .checked_sub(1)
        .ok_or(ScoringError::InvalidRank { rank })? as usize;

    RANK_POINTS
        .get(index)
        .copied()
        .ok_or(ScoringError::InvalidRank { rank })
}

pub fn rating_adjustment(self_rating: i32, opponent_ratings: &[i32]) -> Result<i32, ScoringError> {
    if opponent_ratings.is_empty() {
        return Err(ScoringError::NotEnoughOpponentRatings);
    }

    let sum: i32 = opponent_ratings.iter().sum();
    let average_others = sum / opponent_ratings.len() as i32;

    Ok((average_others - self_rating) / 40)
}

pub fn score(rank: u8, self_rating: i32, opponent_ratings: &[i32]) -> Result<i32, ScoringError> {
    let rank_points = rank_points(rank)?;
    let adjustment = rating_adjustment(self_rating, opponent_ratings)?;
    let raw_score = rank_points + adjustment;

    if rank <= 3 && raw_score <= 0 {
        Ok(1)
    } else {
        Ok(raw_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_points_match_six_player_table() {
        assert_eq!(rank_points(1), Ok(35));
        assert_eq!(rank_points(2), Ok(21));
        assert_eq!(rank_points(3), Ok(7));
        assert_eq!(rank_points(4), Ok(-7));
        assert_eq!(rank_points(5), Ok(-21));
        assert_eq!(rank_points(6), Ok(-35));
    }

    #[test]
    fn rank_outside_six_player_table_is_invalid() {
        assert_eq!(rank_points(0), Err(ScoringError::InvalidRank { rank: 0 }));
        assert_eq!(rank_points(7), Err(ScoringError::InvalidRank { rank: 7 }));
    }

    #[test]
    fn rating_adjustment_truncates_toward_zero() {
        let opponents = [1480, 1480, 1480, 1480, 1480];

        assert_eq!(rating_adjustment(1500, &opponents), Ok(0));
        assert_eq!(rating_adjustment(1519, &opponents), Ok(0));
        assert_eq!(rating_adjustment(1560, &opponents), Ok(-2));
    }

    #[test]
    fn score_adds_rank_points_and_rating_adjustment() {
        let opponents = [1600, 1600, 1600, 1600, 1600];

        assert_eq!(score(1, 1500, &opponents), Ok(37));
        assert_eq!(score(3, 1500, &opponents), Ok(9));
        assert_eq!(score(6, 1500, &opponents), Ok(-33));
    }

    #[test]
    fn top_three_non_positive_scores_are_clamped_to_one() {
        let opponents = [1500, 1500, 1500, 1500, 1500];

        assert_eq!(score(3, 1780, &opponents), Ok(1));
        assert_eq!(score(2, 2340, &opponents), Ok(1));
        assert_eq!(score(1, 2900, &opponents), Ok(1));
    }

    #[test]
    fn bottom_three_non_positive_scores_are_not_clamped() {
        let opponents = [1500, 1500, 1500, 1500, 1500];

        assert_eq!(score(4, 1500, &opponents), Ok(-7));
        assert_eq!(score(5, 1500, &opponents), Ok(-21));
        assert_eq!(score(6, 1500, &opponents), Ok(-35));
    }
}
