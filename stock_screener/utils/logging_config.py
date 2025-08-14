#!/usr/bin/env python3
"""
Centralized Logging Configuration Module
========================================

This module provides a centralized logging configuration system for the entire
Stock Screener application. It creates consistent logging formats, handlers,
and log levels across all modules.

Features:
- Centralized logging configuration
- File and console output
- Rotating log files to prevent disk space issues
- Structured log format with timestamps and module names
- Performance tracking for execution flow
- Error tracking and debugging support

Usage:
    from stock_screener.utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Application started")
    logger.debug("Debug information")
    logger.error("An error occurred")
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class StockScreenerLogger:
    """
    Centralized logger configuration for the Stock Screener application.
    Provides consistent logging setup across all modules.
    """
    
    _initialized = False
    _loggers = {}
    
    @classmethod
    def setup_logging(cls, log_level=logging.INFO):
        """
        Initialize the logging system with consistent configuration.
        
        Args:
            log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if cls._initialized:
            return
            
        # Create logs directory
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger("stock_screener")
        root_logger.setLevel(log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        log_file = log_dir / f"stock_screener_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        
        # Console handler with colors (simplified)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        cls._initialized = True
        
        # Log initialization
        init_logger = cls.get_logger("logging_config")
        init_logger.info("=" * 80)
        init_logger.info("Stock Screener Logging System Initialized")
        init_logger.info(f"Log Level: {logging.getLevelName(log_level)}")
        init_logger.info(f"Log File: {log_file}")
        init_logger.info("=" * 80)
    
    @classmethod
    def get_logger(cls, name):
        """
        Get a logger instance for a specific module.
        
        Args:
            name (str): Logger name (typically __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            # Ensure name starts with stock_screener for hierarchy
            if not name.startswith("stock_screener"):
                if name == "__main__":
                    name = "stock_screener.main"
                else:
                    name = f"stock_screener.{name}"
            
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def log_execution_start(cls, module_name, function_name, **kwargs):
        """
        Log the start of a major execution with parameters.
        
        Args:
            module_name (str): Name of the module
            function_name (str): Name of the function
            **kwargs: Function parameters to log
        """
        logger = cls.get_logger(module_name)
        params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "no parameters"
        logger.info(f"🚀 EXECUTION START: {function_name}({params_str})")
    
    @classmethod
    def log_execution_end(cls, module_name, function_name, duration=None, result_summary=None):
        """
        Log the end of a major execution with results.
        
        Args:
            module_name (str): Name of the module
            function_name (str): Name of the function
            duration (float): Execution duration in seconds
            result_summary (str): Summary of results
        """
        logger = cls.get_logger(module_name)
        duration_str = f" in {duration:.2f}s" if duration else ""
        result_str = f" | Result: {result_summary}" if result_summary else ""
        logger.info(f"✅ EXECUTION END: {function_name}{duration_str}{result_str}")
    
    @classmethod
    def log_progress(cls, module_name, current, total, item_name="items"):
        """
        Log progress of long-running operations.
        
        Args:
            module_name (str): Name of the module
            current (int): Current progress
            total (int): Total items
            item_name (str): Description of items being processed
        """
        logger = cls.get_logger(module_name)
        percentage = (current / total * 100) if total > 0 else 0
        logger.info(f"📊 PROGRESS: {current}/{total} {item_name} ({percentage:.1f}%)")


# Convenience function for easy import
def get_logger(name):
    """
    Convenience function to get a logger instance.
    
    Args:
        name (str): Logger name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return StockScreenerLogger.get_logger(name)


# Convenience functions for execution tracking
def log_execution_start(module_name, function_name, **kwargs):
    """Log the start of major execution."""
    StockScreenerLogger.log_execution_start(module_name, function_name, **kwargs)


def log_execution_end(module_name, function_name, duration=None, result_summary=None):
    """Log the end of major execution."""
    StockScreenerLogger.log_execution_end(module_name, function_name, duration, result_summary)


def log_progress(module_name, current, total, item_name="items"):
    """Log progress of long-running operations."""
    StockScreenerLogger.log_progress(module_name, current, total, item_name)


# Initialize logging when module is imported
if __name__ != "__main__":
    StockScreenerLogger.setup_logging()
