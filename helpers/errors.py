class InvalidParametersError(Exception):
    def __init__(self, message="Invalid parameters were provided."):
        super().__init__(message)
