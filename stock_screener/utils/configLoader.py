#!/usr/bin/env python3
"""
Configuration Loading Module
============================

This module provides utilities to load stock symbols from configuration files.
It handles configuration parsing, validation, and provides fallback mechanisms
for robust configuration management.

Features:
- Configuration file parsing with error handling
- Sector-based symbol loading
- Validation of configuration parameters
- Comprehensive logging for debugging
- Fallback mechanisms for missing configurations

Usage:
    from stock_screener.utils.configLoader import load_symbols_from_config
    
    symbols = load_symbols_from_config()
"""

from configparser import ConfigParser
from pathlib import Path
import os
from stock_screener.utils.tickerLoader import load_tickers
from stock_screener.utils.logging_config import get_logger, log_execution_start, log_execution_end
import time

# Initialize module logger
logger = get_logger(__name__)


def _get_config_path():
    """
    Get the path to the configuration file.
    
    Returns:
        Path: Path to the screener_config.properties file
    """
    config_path = Path(__file__).parent.parent.parent / 'config' / 'screener_config.properties'
    logger.debug(f"Configuration file path: {config_path}")
    return config_path


def _load_configuration():
    """
    Load configuration from the properties file with error handling.
    
    Returns:
        ConfigParser: Loaded configuration object
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        Exception: If configuration parsing fails
    """
    config = ConfigParser()
    config_path = _get_config_path()
    
    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config.read(config_path)
        logger.info(f"Configuration loaded successfully from: {config_path}")
        
        # Log available sections and options
        sections = config.sections()
        if 'DEFAULT' in config:
            sections.insert(0, 'DEFAULT')
        
        logger.debug(f"Available configuration sections: {sections}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def _validate_sector(sector):
    """
    Validate that the sector parameter is valid.
    
    Args:
        sector (str): Sector name to validate
        
    Returns:
        bool: True if sector is valid, False otherwise
    """
    if not sector:
        logger.error("Sector not specified in configuration")
        return False
    
    if not isinstance(sector, str):
        logger.error(f"Sector must be a string, got {type(sector)}: {sector}")
        return False
    
    logger.debug(f"Sector validation passed for: {sector}")
    return True


def load_symbols_from_config():
    """
    Load stock symbols for a given sector from the configuration file.
    
    This function reads the sector from the configuration file and loads
    the corresponding ticker symbols using the tickerLoader module.

    Returns:
        list: List of ticker symbols for the configured sector.

    Raises:
        ValueError: If the sector is not specified in the config or is invalid.
        FileNotFoundError: If the configuration file is not found.
        Exception: If there's an error loading the configuration or tickers.
    """
    start_time = time.time()
    log_execution_start(__name__, "load_symbols_from_config")
    
    try:
        # Load configuration
        config = _load_configuration()
        
        # Get sector from configuration
        sector = config.get('DEFAULT', 'sector', fallback=None)
        logger.info(f"Found sector in configuration: {sector}")
        
        # Validate sector
        if not _validate_sector(sector):
            raise ValueError(f"Invalid or missing sector in configuration: {sector}")
        
        # Load tickers for the specified sector
        logger.info(f"Loading tickers for sector: {sector}")
        symbols = load_tickers(sector)
        
        if not symbols:
            logger.warning(f"No symbols loaded for sector: {sector}")
        else:
            logger.info(f"Successfully loaded {len(symbols)} symbols for sector: {sector}")
            logger.debug(f"Loaded symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
        
        duration = time.time() - start_time
        log_execution_end(__name__, "load_symbols_from_config", duration, f"Loaded {len(symbols)} symbols")
        
        return symbols
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to load symbols from configuration: {e}")
        log_execution_end(__name__, "load_symbols_from_config", duration, f"Failed: {str(e)}")
        raise


def get_config_value(section, key, fallback=None):
    """
    Get a specific configuration value with fallback support.
    
    Args:
        section (str): Configuration section name
        key (str): Configuration key name
        fallback: Fallback value if key is not found
        
    Returns:
        str: Configuration value or fallback
    """
    try:
        config = _load_configuration()
        value = config.get(section, key, fallback=fallback)
        logger.debug(f"Configuration value for [{section}].{key}: {value}")
        return value
    except Exception as e:
        logger.error(f"Failed to get configuration value [{section}].{key}: {e}")
        return fallback


def list_available_sectors():
    """
    List all available sectors that can be configured.
    
    Returns:
        list: List of available sector identifiers
    """
    from stock_screener.core.constants import SCOPE_URL_MAP
    
    sectors = list(SCOPE_URL_MAP.keys())
    logger.info(f"Available sectors: {sectors}")
    return sectors


# Module initialization log
logger.info("Configuration loader module initialized")