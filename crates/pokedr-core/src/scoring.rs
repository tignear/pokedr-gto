#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringError {
    InvalidRank { rank: u8 },
    NotEnoughOpponentRatings,
}

pub const PLAYER_COUNT: usize = 6;

const RANK_POINTS: [i32; PLAYER_COUNT] = [40, 15, 3, 0, -18, -40];

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
        assert_eq!(rank_points(1), Ok(40));
        assert_eq!(rank_points(2), Ok(15));
        assert_eq!(rank_points(3), Ok(3));
        assert_eq!(rank_points(4), Ok(0));
        assert_eq!(rank_points(5), Ok(-18));
        assert_eq!(rank_points(6), Ok(-40));
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

        assert_eq!(score(1, 1500, &opponents), Ok(42));
        assert_eq!(score(3, 1500, &opponents), Ok(5));
        assert_eq!(score(6, 1500, &opponents), Ok(-38));
    }

    #[test]
    fn top_three_non_positive_scores_are_clamped_to_one() {
        let opponents = [1500, 1500, 1500, 1500, 1500];

        assert_eq!(score(3, 1620, &opponents), Ok(1));
        assert_eq!(score(2, 2100, &opponents), Ok(1));
        assert_eq!(score(1, 3100, &opponents), Ok(1));
    }

    #[test]
    fn bottom_three_non_positive_scores_are_not_clamped() {
        let opponents = [1500, 1500, 1500, 1500, 1500];

        assert_eq!(score(4, 1500, &opponents), Ok(0));
        assert_eq!(score(5, 1500, &opponents), Ok(-18));
        assert_eq!(score(6, 1500, &opponents), Ok(-40));
    }
}
